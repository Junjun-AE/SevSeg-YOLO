<div align="center">

# SevSeg-YOLO

**Unified Detection, Severity Scoring & Zero-Annotation Segmentation for Industrial Defects**

<br>

[![Python](https://img.shields.io/badge/python-≥3.8-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-≥1.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](CHANGELOG.md)

One model &nbsp;·&nbsp; Three tasks &nbsp;·&nbsp; Single forward pass &nbsp;·&nbsp; No mask annotations needed

<br>

[English](README.md) &nbsp;|&nbsp; [中文文档](README_zh.md) &nbsp;|&nbsp; [Contributing](CONTRIBUTING.md) &nbsp;|&nbsp; [Changelog](CHANGELOG.md)

</div>

<br>

## Why SevSeg-YOLO?

On a high-speed inspection line scanning **60+ products/min**, every defect needs three answers:

| Question | Traditional Approach | Pain Point |
|:---|:---|:---|
| **Is there a defect?** | Standard YOLO detector | ✅ Solved |
| **How severe?** (grade 2 = accept, grade 8 = scrap) | Separate classifier | Extra model + latency |
| **What area does it cover?** (ISO/GB compliance) | Instance segmentation | 10–20× annotation cost |

**SevSeg-YOLO answers all three in one model, one pass** — the detection head directly predicts a continuous severity score, while a training-free MaskGenerator derives approximate binary masks from FPN features.

<br>

## Highlights

**Gaussian NLL Severity Head** — Models the ±1 subjectivity of human inspectors as observation noise. Compared to Smooth L1 (5-seed average): **MAE ↓ 21.2%**, **Spearman ρ ↑ 54.3%**.

**MaskGenerator (Zero-Annotation)** — Pure CPU post-processing that converts implicit defect knowledge in FPN features into explicit binary masks via bimodal channel selection + Canny-guided upsampling. **100% mask validity**, **1.13 ms** median latency.

**Real-Time Deployment** — All 5 model scales end-to-end **< 10 ms** on TensorRT FP16 (**> 100 FPS**). Nano scale: **534 FPS** pure inference, **7.4 MB** engine.

<br>

## Quick Start

### Installation

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e .
```

> Only standard `opencv-python` is required — no `opencv-contrib`.

Optional extras for export and deployment:

```bash
pip install -e ".[export]"      # ONNX export
pip install -e ".[tensorrt]"    # TensorRT deployment
```

### Inference in 3 Lines

```python
from sevseg_yolo import SevSegYOLO

model = SevSegYOLO("best.pt")
result = model.predict("image.jpg")

for det in result.detections:
    print(f"{det.class_name}: severity={det.severity:.1f}, mask_fill={det.fill_ratio:.3f}")
```

<br>

## Full Workflow

### Step 1 · Prepare Dataset

<details>
<summary><b>1.1 &nbsp; Annotate with LabelMe</b></summary>
<br>

Use [LabelMe](https://github.com/wkentaro/labelme) to draw rectangle bounding boxes. Write severity in the `description` field:

```json
{
  "shapes": [{
    "label": "scratch",
    "points": [[120, 80], [250, 180]],
    "shape_type": "rectangle",
    "description": "severity=7.5"
  }]
}
```

Scoring guideline:

| Severity | Meaning | Action |
|:---:|:---|:---|
| 0 | No defect | — |
| 1 – 3 | Minor | Accept |
| 4 – 6 | Moderate | Rework |
| 7 – 10 | Severe | Scrap |

</details>

<details>
<summary><b>1.2 &nbsp; Organize directories</b></summary>
<br>

```
my_dataset/
├── images/
│   ├── img_001.jpg
│   └── ...
└── jsons/
    ├── img_001.json          ← LabelMe annotation files
    └── ...
```

</details>

<details>
<summary><b>1.3 &nbsp; Convert to YOLO format</b></summary>
<br>

```python
from sevseg_yolo.convert import convert_dataset

convert_dataset(
    images_dir="my_dataset/images",
    jsons_dir="my_dataset/jsons",
    output_dir="my_dataset_yolo",
    val_ratio=0.2,
)
```

Produces standard YOLO layout with 6-column labels: `class_id cx cy w h severity`.

</details>

### Step 2 · Train

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26m-score.yaml")
model.train(
    task="score_detect",
    data="my_dataset_yolo/data.yaml",
    pretrained="yolo26m.pt",
    epochs=105, batch=32, imgsz=640,
    mixup=0.0,   # ⚠️ MUST be 0 — severity cannot be interpolated across mixed images
)
```

> **Tip:** Start with `n` (nano) scale for quick validation → switch to `m` or `l` → monitor `score_loss` convergence.

### Step 3 · Inference

<details>
<summary><b>Python API</b> (recommended)</summary>
<br>

```python
from sevseg_yolo import SevSegYOLO

model = SevSegYOLO("runs/score_detect/train/weights/best.pt")
result = model.predict("test.jpg")

for det in result.detections:
    print(f"Class: {det.class_name}, Severity: {det.severity:.1f}/10")
    print(f"  Bbox: {det.bbox}, Mask: {det.mask.shape}, Fill: {det.fill_ratio:.3f}")

# Visualize and save
result.visualize().save("output.jpg")

# Filter by severity / confidence
severe = result.filter(min_severity=7.0, min_confidence=0.5)

# Batch inference
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])

# JSON-serializable output
data = result.to_dict()
```

</details>

<details>
<summary><b>CLI</b></summary>
<br>

```bash
python tools/predict_demo.py --weights best.pt --source test_images/ --save-dir outputs/
```

</details>

### Step 4 · Export & Deploy

<details>
<summary><b>ONNX + TensorRT</b></summary>
<br>

```python
from sevseg_yolo.export import export_scoreyolo_onnx
from sevseg_yolo.tensorrt_deploy import deploy_scoreyolo

# Export to ONNX (with optional PCA feature compression)
export_scoreyolo_onnx(model.model, "model.onnx", imgsz=640, opset=17)

# Build TensorRT engine
deploy_scoreyolo("model.onnx", "model.engine", fp16=True, max_batch=4)
```

ONNX output format: `det_output (B, K, 7)` → `[x1, y1, x2, y2, conf, cls, severity]`, with optional `feat_p3/p4/p5` nodes for MaskGenerator.

</details>

### Step 5 · Evaluate

```python
from sevseg_yolo.evaluation import full_score_evaluation, print_evaluation_report

metrics = full_score_evaluation(pred_scores, gt_scores)
print_evaluation_report(metrics)
# → MAE, Spearman ρ, ±1 tolerance accuracy, low/high-end misjudge rates,
#   segment-wise MAE, 11×11 confusion matrix
```

<br>

## MaskGenerator

A **pure post-processing module** — no training required, runs on CPU. Converts implicit defect knowledge in FPN features into explicit binary masks through a 6-step pipeline:

1. **Scale-adaptive feature selection** — choose P3/P4/P5 based on bbox size
2. **Bimodal Top-K channel selection** — pick channels with strongest defect-vs-normal separation
3. **Multi-scale weighted fusion** — combine selected channels into a single activation map
4. **Canny edge-guided upsampling** — upsample activation using original image edges as guidance
5. **Adaptive binarization** — local thresholding to produce binary mask
6. **Morphological refinement** — close + open to remove noise and fill gaps

**Why bimodal selection (V3)?** Variance-based selection (V2) may pick channels with high spatial variance from background texture. Bimodal selection measures the gap between the top-30% and bottom-30% pixel intensities within each bbox — directly quantifying defect-vs-normal separation.

```python
model = SevSegYOLO("best.pt", mask_version="v3")  # bimodal (default)
model = SevSegYOLO("best.pt", mask_version="v2")  # variance (legacy)
model = SevSegYOLO("best.pt", mask_enabled=False)  # detection + score only
```

<br>

## Model Zoo

All results are 5-seed averages with Gaussian NLL (σ = 0.1, λ = 0.05):

| Scale | Params | mAP@50 | Score MAE ↓ | Spearman ρ ↑ |
|:---:|:---:|:---:|:---:|:---:|
| **n** (nano) | 2.57 M | 0.513 | 1.317 | 0.742 |
| **s** (small) | 10.19 M | 0.573 | 1.306 | 0.720 |
| **m** (medium) | 22.19 M | 0.608 | 1.316 | 0.715 |
| **l** (large) | 26.59 M | 0.626 | 1.297 | 0.709 |
| **x** (xlarge) | 56.08 M | 0.623 | 1.224 | 0.744 |

Model configs: `ultralytics/cfg/models/26/yolo26{n,s,m,l,x}-score.yaml`

<br>

## Project Structure

```
sevseg-yolo/
├── sevseg_yolo/                  # Core package
│   ├── model.py                  # SevSegYOLO — unified inference entry point
│   ├── mask_generator_v3.py      # V3: bimodal channel selection (default)
│   ├── mask_generator_v2.py      # V2: variance channel selection (legacy)
│   ├── convert.py                # LabelMe JSON → 6-column YOLO format
│   ├── evaluation.py             # Severity metrics (MAE, Spearman ρ, tolerance, etc.)
│   ├── export.py                 # ONNX export with optional PCA compression
│   ├── tensorrt_deploy.py        # TensorRT FP16/INT8 deployment pipeline
│   ├── visualization.py          # Scatter plots, rank curves, confusion heatmaps
│   └── utils.py                  # Feature hooks, coordinate helpers
│
├── ultralytics/                  # Modified Ultralytics (YOLO26 + ScoreDetect)
│   ├── nn/modules/head.py        # ScoreHead & ScoreDetect head
│   ├── utils/loss.py             # Gaussian NLL score loss
│   └── cfg/models/26/            # yolo26{n,s,m,l,x}-score.yaml
│
├── configs/                      # Training configuration templates
├── tools/                        # CLI scripts (predict_demo, visualize_masks)
└── pyproject.toml
```

<br>

## Important Notes

| Item | Detail |
|:---|:---|
| **MixUp = 0** | Mandatory — severity scores cannot be interpolated across mixed images |
| **Severity range** | 0.0 – 10.0 (internally normalized to 0 – 1) |
| **MaskGenerator** | Approximate feature-based masks, not pixel-perfect ground truth |
| **Gaussian σ = 0.1** | Derived from ±1 inter-annotator noise |
| **OpenCV** | Standard `opencv-python` only, `opencv-contrib` is **not** needed |

<br>

## Citation

```bibtex
@software{sevseg_yolo_2026,
  title  = {SevSeg-YOLO: Unified Detection, Severity Scoring, and
            Annotation-Free Approximate Segmentation for Industrial Defects},
  author = {SevSeg-YOLO Contributors},
  year   = {2026},
  url    = {https://github.com/sevseg-yolo/sevseg-yolo}
}
```

## License

[AGPL-3.0](LICENSE). Modified Ultralytics code retains its original AGPL-3.0 license.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO framework
- [LabelMe](https://github.com/wkentaro/labelme) — Annotation tool
