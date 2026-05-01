"""
YOLO comparison on Modal: from-scratch vs. transfer-learning.

Run:
    modal run yolo_comparison_modal.py

What it does (all on a Modal GPU container):
    1. Downloads COCO128 (Ultralytics auto-pulls it on first train()).
    2. Trains yolov8n from scratch (yolov8n.yaml, no pretrained weights).
    3. Trains yolov8n with transfer learning (yolov8n.pt).
    4. Validates both, reports mAP/precision/recall.
    5. Runs inference on a fresh test image.
    6. Returns the two annotated images + a side-by-side comparison and
       writes them to ./outputs/ on your local machine.

Override settings via CLI:
    modal run yolo_comparison_modal.py --epochs 80 --imgsz 640 --batch 32

Use your own test image (URL or local path):
    modal run yolo_comparison_modal.py --test-image ./my_image.jpg
"""

from __future__ import annotations

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
                         test_image: bytes, data: str, imgsz: int) -> dict:
    """Run validation + inference on the same image with both models."""
    import matplotlib.pyplot as plt
    from PIL import Image

    scratch_metrics = _validate(scratch_pt, data, imgsz)
    transfer_metrics = _validate(transfer_pt, data, imgsz)

    scratch_img, scratch_det = _predict(scratch_pt, test_image, imgsz)
    transfer_img, transfer_det = _predict(transfer_pt, test_image, imgsz)

    # Side-by-side comparison figure.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(Image.open(io.BytesIO(scratch_img)))
    axes[0].set_title("From scratch"); axes[0].axis("off")
    axes[1].imshow(Image.open(io.BytesIO(transfer_img)))
    axes[1].set_title("Transfer learning"); axes[1].axis("off")
    plt.tight_layout()
    cmp_buf = io.BytesIO()
    plt.savefig(cmp_buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "scratch": {"metrics": scratch_metrics, "detections": scratch_det,
                     "image_png": scratch_img},
        "transfer": {"metrics": transfer_metrics, "detections": transfer_det,
                      "image_png": transfer_img},
        "comparison_png": cmp_buf.getvalue(),
    }


def _load_test_image(test_image: str | None) -> bytes:
    import urllib.request
    if test_image is None:
        url = "https://ultralytics.com/images/bus.jpg"
        with urllib.request.urlopen(url) as r:
            return r.read()
    p = Path(test_image)
    if p.exists():
        return p.read_bytes()
    with urllib.request.urlopen(test_image) as r:
        return r.read()


@app.local_entrypoint()
def main(epochs: int = 50,
         imgsz: int = 640,
         batch: int = 16,
         data: str = "coco128.yaml",
         test_image: str | None = None,
         out_dir: str = "outputs"):
    """Orchestrates training, eval, inference. Writes results to ./outputs/."""
    test_bytes = _load_test_image(test_image)

    print(f"Training (epochs={epochs}, imgsz={imgsz}, batch={batch}, data={data})")
    # Train both models in parallel — independent GPU containers.
    scratch_call = train_from_scratch.spawn(data, epochs, imgsz, batch)
    transfer_call = train_transfer.spawn(data, epochs, imgsz, batch)
    scratch_pt = scratch_call.get()
    transfer_pt = transfer_call.get()
    print(f"  scratch best : {scratch_pt}")
    print(f"  transfer best: {transfer_pt}")

    print("Evaluating + running inference on test image")
    result = evaluate_and_predict.remote(scratch_pt, transfer_pt,
                                          test_bytes, data, imgsz)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "scratch.png").write_bytes(result["scratch"]["image_png"])
    (out / "transfer.png").write_bytes(result["transfer"]["image_png"])
    (out / "comparison.png").write_bytes(result["comparison_png"])
    (out / "test_input.jpg").write_bytes(test_bytes)

    print("\n=== Validation metrics ===")
    fmt = "{:<22}{:>10}{:>12}{:>10}{:>10}".format
    print(fmt("model", "mAP@50", "mAP@50-95", "P", "R"))
    for label, key in [("From scratch", "scratch"), ("Transfer learning", "transfer")]:
        m = result[key]["metrics"]
        print(f"{label:<22}{m['mAP50']:>10.3f}{m['mAP50-95']:>12.3f}"
              f"{m['precision']:>10.3f}{m['recall']:>10.3f}")

    for label, key in [("From scratch", "scratch"), ("Transfer learning", "transfer")]:
        dets = result[key]["detections"]
        print(f"\n--- {label}: {len(dets)} detections ---")
        for d in dets:
            print(f"  {d['class']:<15} conf={d['conf']:.3f}  bbox={d['xyxy']}")

    print(f"\nWrote results to {out.resolve()}/")
