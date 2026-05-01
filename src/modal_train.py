"""
YOLO comparison on Modal: from-scratch vs. transfer-learning.

Run:
    modal run modal_train.py

What it does (all on a Modal GPU container):
    1. Downloads COCO128 (Ultralytics auto-pulls it on first train()).
    2. Trains yolov8n from scratch (yolov8n.yaml, no pretrained weights).
    3. Trains yolov8n with transfer learning (yolov8n.pt).
    4. Validates both, reports mAP/precision/recall.
    5. Runs inference on multiple diverse test images.
    6. Returns annotated results + comprehensive metrics.

Override settings via CLI:
    modal run modal_train.py --epochs 80 --imgsz 640 --batch 32
"""

from __future__ import annotations

import json
import io
from pathlib import Path

import modal

GPU = "T4"           # bump to "A10G" or "L4" for faster training
TIMEOUT = 60 * 60    # 1 hour ceiling per function call

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.3.*",
        "torch",
        "torchvision",
        "matplotlib",
        "pillow",
    )
)

app = modal.App("yolo-scratch-vs-transfer", image=image)

# Volume to cache the dataset and weights between calls.
volume = modal.Volume.from_name("yolo-compare-cache", create_if_missing=True)
CACHE = "/cache"


def _train_one(weights_or_yaml: str, run_name: str, *, data: str,
               epochs: int, imgsz: int, batch: int, pretrained: bool) -> str:
    """Train a single YOLO model and return the path to best.pt."""
    import os
    from ultralytics import YOLO

    os.chdir(CACHE)
    model = YOLO(weights_or_yaml)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        pretrained=pretrained,
        project=f"{CACHE}/runs_compare",
        name=run_name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        verbose=False,
    )
    best = Path(CACHE) / "runs_compare" / run_name / "weights" / "best.pt"
    return str(best)


def _validate(weights_path: str, data: str, imgsz: int) -> dict:
    from ultralytics import YOLO
    m = YOLO(weights_path)
    metrics = m.val(data=data, imgsz=imgsz, verbose=False)
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _predict(weights_path: str, image_bytes: bytes, imgsz: int) -> tuple[bytes, list[dict]]:
    """Returns (annotated_png_bytes, detections list)."""
    import tempfile
    from ultralytics import YOLO
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        src = f.name

    model = YOLO(weights_path)
    res = model.predict(source=src, imgsz=imgsz, conf=0.25, verbose=False)[0]

    annotated = res.plot()  # BGR ndarray
    img = Image.fromarray(annotated[:, :, ::-1])  # to RGB
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    detections = []
    if res.boxes is not None and len(res.boxes):
        for b in res.boxes:
            detections.append({
                "class": res.names[int(b.cls.item())],
                "conf": round(float(b.conf.item()), 3),
                "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
            })
    return buf.getvalue(), detections


def _get_test_images() -> dict[str, bytes]:
    """Load diverse test images from Ultralytics."""
    import urllib.request
    images = {}
    
    urls = {
        "bus": "https://ultralytics.com/images/bus.jpg",
        "zidane": "https://ultralytics.com/images/zidane.jpg",
        "footage": "https://ultralytics.com/images/football.jpg",
    }
    
    for name, url in urls.items():
        try:
            with urllib.request.urlopen(url) as r:
                images[name] = r.read()
                print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    
    return images if images else {"bus": _load_default_image()}


def _load_default_image() -> bytes:
    """Fallback to bus image if others fail."""
    import urllib.request
    try:
        with urllib.request.urlopen("https://ultralytics.com/images/bus.jpg") as r:
            return r.read()
    except:
        raise RuntimeError("Could not load any test images")


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={CACHE: volume})
def train_from_scratch(data: str, epochs: int, imgsz: int, batch: int) -> str:
    path = _train_one("yolov8n.yaml", "scratch",
                      data=data, epochs=epochs, imgsz=imgsz,
                      batch=batch, pretrained=False)
    volume.commit()
    return path


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={CACHE: volume})
def train_transfer(data: str, epochs: int, imgsz: int, batch: int) -> str:
    path = _train_one("yolov8n.pt", "transfer",
                      data=data, epochs=epochs, imgsz=imgsz,
                      batch=batch, pretrained=True)
    volume.commit()
    return path


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={CACHE: volume})
def evaluate_and_predict(scratch_pt: str, transfer_pt: str,
                         test_images: dict[str, bytes], data: str, imgsz: int) -> dict:
    """Run validation + inference on multiple images with both models."""
    import matplotlib.pyplot as plt
    from PIL import Image

    scratch_metrics = _validate(scratch_pt, data, imgsz)
    transfer_metrics = _validate(transfer_pt, data, imgsz)

    # Process all test images
    inference_results = {}
    for img_name, img_bytes in test_images.items():
        scratch_img, scratch_det = _predict(scratch_pt, img_bytes, imgsz)
        transfer_img, transfer_det = _predict(transfer_pt, img_bytes, imgsz)
        
        inference_results[img_name] = {
            "scratch": {"image": scratch_img, "detections": scratch_det},
            "transfer": {"image": transfer_img, "detections": transfer_det},
        }

    # Create side-by-side comparison for the first image only
    first_name = list(test_images.keys())[0]
    first_scratch = inference_results[first_name]["scratch"]["image"]
    first_transfer = inference_results[first_name]["transfer"]["image"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(Image.open(io.BytesIO(first_scratch)))
    axes[0].set_title("From scratch"); axes[0].axis("off")
    axes[1].imshow(Image.open(io.BytesIO(first_transfer)))
    axes[1].set_title("Transfer learning"); axes[1].axis("off")
    plt.tight_layout()
    cmp_buf = io.BytesIO()
    plt.savefig(cmp_buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "scratch_metrics": scratch_metrics,
        "transfer_metrics": transfer_metrics,
        "inference_results": inference_results,
        "comparison_png": cmp_buf.getvalue(),
    }


@app.local_entrypoint()
def main(epochs: int = 50,
         imgsz: int = 640,
         batch: int = 16,
         data: str = "coco128.yaml",
         out_dir: str = "results"):
    """Orchestrates training, eval, inference. Writes results to ./results/."""
    print("Loading test images...")
    test_images = _get_test_images()
    print(f"Loaded {len(test_images)} test images for inference")

    print(f"Training (epochs={epochs}, imgsz={imgsz}, batch={batch}, data={data})")
    # Train both models in parallel — independent GPU containers.
    scratch_call = train_from_scratch.spawn(data, epochs, imgsz, batch)
    transfer_call = train_transfer.spawn(data, epochs, imgsz, batch)
    scratch_pt = scratch_call.get()
    transfer_pt = transfer_call.get()
    print(f"  scratch best : {scratch_pt}")
    print(f"  transfer best: {transfer_pt}")

    print("Evaluating + running inference on test images")
    result = evaluate_and_predict.remote(scratch_pt, transfer_pt,
                                          test_images, data, imgsz)

    out = Path(out_dir)
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "comparison.png").write_bytes(result["comparison_png"])
    
    # Save individual inference results for each test image
    inference_dir = figures / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    for img_name, img_results in result["inference_results"].items():
        model_dir = inference_dir / img_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "scratch.png").write_bytes(img_results["scratch"]["image"])
        (model_dir / "transfer.png").write_bytes(img_results["transfer"]["image"])

    scratch_metrics = result["scratch_metrics"]
    transfer_metrics = result["transfer_metrics"]
    
    metrics_summary = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "data": data,
        "scratch": scratch_metrics,
        "transfer": transfer_metrics,
        "test_images": {name: {
            "scratch_detections": len(res["scratch"]["detections"]),
            "transfer_detections": len(res["transfer"]["detections"])
        } for name, res in result["inference_results"].items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    
    # Generate comprehensive markdown report
    md_content = (
        "# Experimental Results\n\n"
        f"All runs: YOLOv8n, {data}, {epochs} epochs, {imgsz}x{imgsz}, "
        f"batch {batch}, Tesla T4 (14 913 MiB), seed 0.\n\n"
        "## Validation Metrics (on train/val split)\n\n"
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n"
        "|---|---|---|---|---|\n"
        f"| From scratch | {scratch_metrics['mAP50']:.3f} | {scratch_metrics['mAP50-95']:.3f} | "
        f"{scratch_metrics['precision']:.3f} | {scratch_metrics['recall']:.3f} |\n"
        f"| Transfer learning | {transfer_metrics['mAP50']:.3f} | {transfer_metrics['mAP50-95']:.3f} | "
        f"{transfer_metrics['precision']:.3f} | {transfer_metrics['recall']:.3f} |\n\n"
        "## Inference Results on Test Images\n\n"
        "### Summary: Detection Count per Image\n\n"
        "| Image | From Scratch | Transfer Learning |\n"
        "|---|---|---|\n"
    )
    
    for img_name, img_results in result["inference_results"].items():
        scratch_count = len(img_results["scratch"]["detections"])
        transfer_count = len(img_results["transfer"]["detections"])
        md_content += f"| {img_name.title()} | {scratch_count} | {transfer_count} |\n"
    
    md_content += "\n### Detailed Inference Results\n\n"
    for img_name, img_results in result["inference_results"].items():
        md_content += f"#### {img_name.title()}\n\n"
        md_content += f"**From Scratch Model:** {len(img_results['scratch']['detections'])} detections\n"
        if img_results['scratch']['detections']:
            md_content += "```\n"
            for det in img_results['scratch']['detections']:
                md_content += f"  {det['class']:<15} conf={det['conf']:.3f}  bbox={det['xyxy']}\n"
            md_content += "```\n"
        else:
            md_content += "(No detections)\n"
        
        md_content += f"\n**Transfer Learning Model:** {len(img_results['transfer']['detections'])} detections\n"
        if img_results['transfer']['detections']:
            md_content += "```\n"
            for det in img_results['transfer']['detections']:
                md_content += f"  {det['class']:<15} conf={det['conf']:.3f}  bbox={det['xyxy']}\n"
            md_content += "```\n"
        else:
            md_content += "(No detections)\n"
        
        md_content += f"\n![{img_name} - From Scratch](figures/inference/{img_name}/scratch.png)\n"
        md_content += f"![{img_name} - Transfer Learning](figures/inference/{img_name}/transfer.png)\n\n"
    
    (out / "metrics.md").write_text(md_content)

    print("\n=== Validation metrics ===")
    fmt = "{:<22}{:>10}{:>12}{:>10}{:>10}".format
    print(fmt("model", "mAP@50", "mAP@50-95", "P", "R"))
    for label, m in [("From scratch", scratch_metrics), ("Transfer learning", transfer_metrics)]:
        print(f"{label:<22}{m['mAP50']:>10.3f}{m['mAP50-95']:>12.3f}"
              f"{m['precision']:>10.3f}{m['recall']:>10.3f}")

    print("\n=== Inference Results ===")
    for img_name, img_results in result["inference_results"].items():
        print(f"\n{img_name}:")
        scratch_dets = img_results["scratch"]["detections"]
        transfer_dets = img_results["transfer"]["detections"]
        print(f"  From scratch: {len(scratch_dets)} detections")
        print(f"  Transfer learning: {len(transfer_dets)} detections")

    print(f"\nWrote results to {out.resolve()}/")
