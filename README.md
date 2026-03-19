<div align="center">

<img src="https://img.shields.io/badge/🔬-SevSeg--YOLO-blue?style=for-the-badge&labelColor=0a0a23" alt="SevSeg-YOLO" height="40">

# SevSeg-YOLO

### Unified Detection, Severity Scoring & Zero-Annotation Segmentation for Industrial Defects

**One model · Three tasks · Single forward pass · No mask annotations**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-AGPL_3.0-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange?style=flat-square)](CHANGELOG.md)

**[📖 中文文档](README_zh.md)** · **English**

---

SevSeg-YOLO extends YOLO26 with a lightweight **ScoreHead** branch (<3% parameter overhead) that predicts **bounding boxes**, continuous **\[0,10\] severity scores**, and approximate **contour masks** — all in a single forward pass, requiring **zero mask annotations**.

</div>

---

## ✨ Why SevSeg-YOLO?

Consider a precision optics inspection line scanning 60+ products per minute. For every defect, the quality pipeline needs three answers:

| Question | Traditional Solution | SevSeg-YOLO |
|:---|:---|:---|
| **Is there a defect?** | YOLO detector ✅ | ✅ Built-in |
| **How severe is it?** (grade 2 = accept, grade 8 = scrap) | Separate classifier (extra model, extra latency) | ✅ ScoreHead outputs 0–10 continuously |
| **What area does it cover?** (ISO/GB compliance) | Pixel-level segmentation (10–20× annotation cost) | ✅ MaskGenerator derives masks from detection features, **zero extra annotations** |

SevSeg-YOLO answers all three in **one model, one forward pass**. No cascading, no extra labels, no extra latency.

### Key Innovations

- 🧪 **Gaussian NLL Loss** — Models the ±1-score subjectivity of human inspectors as observation noise, reducing MAE by 21.2% vs Smooth L1
- 🎭 **MaskGenerator** — Post-processing module that extracts pixel-level masks from FPN features using bimodal channel selection + Canny-guided upsampling. CPU median 1.13ms, 100% mask validity
- ⚡ **Real-Time** — All 5 model scales achieve end-to-end < 10ms (> 100 FPS) on TRT FP16

---

## 🏗️ Architecture

```
  Input Image (640×640)
         │
         ▼
  ┌─────────────────────────┐
  │   YOLO26 Backbone+Neck  │
  │   (CSP-DarkNet + FPN)   │
  │                         │
  │   P3(s=8) P4(s=16) P5  │
  └──────────┬──────────────┘
             │
  ┌──────────▼──────────────┐
  │    ScoreDetect Head     │
  │                         │
  │  ┌─────┐ ┌─────┐ ┌────────────┐
  │  │ Box │ │ Cls │ │ ScoreHead  │ ← NEW (<3% params)
  │  │ Head│ │ Head│ │ DWConv→1×1 │
  │  │     │ │     │ │ →Sigmoid×10│
  │  └──┬──┘ └──┬──┘ └─────┬──────┘
  └─────┼───────┼──────────┼────────┘
        └───┬───┘          │
            ▼              ▼
     Detection         Severity
  (x1,y1,x2,y2,       (0 – 10)
   conf, class)
            │
            ▼
  ┌─────────────────────────┐
  │    MaskGenerator        │  (post-processing, CPU, no training)
  │                         │
  │  1. Scale-adaptive      │
  │     feature selection   │
  │  2. Bimodal Top-K       │
  │     channel selection   │
  │  3. Multi-scale fusion  │
  │  4. Canny edge-guided   │
  │     upsampling          │
  │  5. Adaptive threshold  │
  │  6. Morphology          │
  └──────────┬──────────────┘
             ▼
      Final Output:
  boxes + classes + conf
  + severity + masks
```

---

## 🚀 Installation

### From Source (recommended)

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e .
```

### Dependencies

SevSeg-YOLO requires only standard packages — **no `opencv-contrib` needed**:

```
torch >= 1.8.0
opencv-python >= 4.6.0
numpy, scipy, matplotlib, pillow, pyyaml
```

For ONNX export: `pip install -e ".[export]"`

For TensorRT: `pip install -e ".[tensorrt]"`

---

## 📖 Tutorial: Complete Workflow

### Step 1: Prepare Your Dataset

#### 1.1 Annotate with LabelMe

Use [LabelMe](https://github.com/wkentaro/labelme) to draw **rectangle** bounding boxes around defects. In the `description` field, write the severity score:

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

> 💡 **Severity guidelines**: 0 = no defect, 1–3 = minor (accept), 4–6 = moderate (rework), 7–10 = severe (scrap). Adjust to your quality standards.

#### 1.2 Organize Directory Structure

```
my_dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── jsons/
    ├── img_001.json    ← LabelMe JSON
    ├── img_002.json
    └── ...
```

#### 1.3 Convert to YOLO Format

```python
from sevseg_yolo.convert import convert_dataset

convert_dataset(
    images_dir="my_dataset/images",
    jsons_dir="my_dataset/jsons",
    output_dir="my_dataset_yolo",
    val_ratio=0.2,   # 80% train, 20% val
)
```

This produces:

```
my_dataset_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   │   └── img_001.txt   ← "0 0.500 0.300 0.120 0.080 7.5"
│   └── val/
└── data.yaml              ← auto-generated config
```

**Label format** — standard YOLO + 6th column for severity:

```
# class_id  center_x  center_y  width  height  severity
0  0.500  0.300  0.120  0.080  7.5
1  0.250  0.600  0.050  0.040  3.0
```

### Step 2: Train

```python
from ultralytics import YOLO

# Choose a model scale: n / s / m / l / x
model = YOLO("ultralytics/cfg/models/26/yolo26m-score.yaml")

model.train(
    task="score_detect",
    data="my_dataset_yolo/data.yaml",
    pretrained="yolo26m.pt",       # YOLO26 pretrained backbone
    epochs=105,
    batch=32,
    imgsz=640,
    mixup=0.0,                     # ⚠️ MUST be 0
)
```

> ⚠️ **MixUp must be disabled** — severity scores cannot be linearly interpolated between mixed images. A "severe crack × 0.3 + minor scratch × 0.7 = severity 5.2" has no physical meaning.

**Training tips:**
- Start with the `n` (nano) scale for quick experiments, then scale up
- Use the provided `configs/train_score.yaml` as a starting point
- The Score Head trains from scratch while the backbone loads pretrained weights
- Monitor `score_loss` alongside the standard `box_loss` and `cls_loss`

### Step 3: Inference

#### Python API (recommended)

```python
from sevseg_yolo import SevSegYOLO

# Load model
model = SevSegYOLO("runs/score_detect/train/weights/best.pt")

# Single image
result = model.predict("test_image.jpg")

for det in result.detections:
    print(f"Class: {det.class_name}")
    print(f"  Severity: {det.severity:.1f} / 10")
    print(f"  Confidence: {det.confidence:.2f}")
    print(f"  Bounding box: {det.bbox}")
    print(f"  Mask shape: {det.mask.shape}")
    print(f"  Mask fill ratio: {det.fill_ratio:.3f}")

# Batch inference
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])
```

#### Command Line

```bash
python tools/predict_demo.py \
    --weights best.pt \
    --source test_images/ \
    --save-dir outputs/ \
    --conf 0.15
```

#### Visualization

```python
# Save visualization with boxes, severity labels, and mask overlays
result.visualize().save("output.jpg")
```

### Step 4: Export & Deploy

#### ONNX Export

```python
from sevseg_yolo.export import export_scoreyolo_onnx

export_scoreyolo_onnx(
    model.model,
    "model.onnx",
    imgsz=640,
    opset=17,
)
```

#### TensorRT Deployment

```python
from sevseg_yolo.tensorrt_deploy import deploy_scoreyolo

deploy_scoreyolo(
    "model.onnx",
    "model.engine",
    fp16=True,          # FP16 quantization
    max_batch=4,
)
```

### Step 5: Evaluate Severity Scoring

```python
from sevseg_yolo.evaluation import full_score_evaluation

metrics = full_score_evaluation(pred_scores, gt_scores)
print(f"MAE: {metrics['mae']:.3f}")
print(f"Spearman ρ: {metrics['spearman_rho']:.3f}")
print(f"±1 Tolerance Accuracy: {metrics['tolerance_acc']:.1%}")
```

---

## 🎭 MaskGenerator: How It Works

MaskGenerator is a **pure post-processing module** — it does not participate in training and runs entirely on CPU. It leverages the FPN features (P3/P4/P5) that the detection model already produces to derive pixel-level defect masks.

**Core insight**: A backbone trained for defect detection implicitly learns to distinguish defects from normal regions in its feature maps. MaskGenerator converts this implicit knowledge into explicit binary masks.

| Step | Operation | Purpose |
|:---:|:---|:---|
| 1 | **Scale-adaptive feature selection** | Choose best feature combination based on model scale (n/s: Layer2+P3+P4; m/l/x: P3+P4+P5) |
| 2 | **Bimodal Top-K channel selection** | For each channel in the bbox crop, measure the gap between the mean of the darkest 30% and brightest 30% of pixels. Select channels with the largest gap — these best separate defect from normal. |
| 3 | **Multi-scale weighted fusion** | Upsample activations from each scale to the highest resolution, then weighted average |
| 4 | **Canny edge-guided upsampling** | Use original image edges to guide the low→high resolution upsampling. Edge zones keep sharp activation, smooth zones get denoised. |
| 5 | **Adaptive binarization** | Convert continuous activation to binary mask via `adaptiveThreshold` |
| 6 | **Morphology** | Close (fill holes) → Open (remove noise dots) → final {0, 1} binary mask |

---

## 📊 Model Zoo

| Scale | Params | mAP@50 | MAE | Spearman ρ |
|:---:|:---:|:---:|:---:|:---:|
| **n** | 2.57M | 0.513 | 1.317 | 0.742 |
| **s** | 10.19M | 0.573 | 1.306 | 0.720 |
| **m** | 22.19M | 0.608 | 1.316 | 0.715 |
| **l** | 26.59M | 0.626 | 1.297 | 0.709 |
| **x** | 56.08M | 0.623 | 1.224 | 0.744 |

> Results from 5-seed experiments with Gaussian NLL (σ=0.1, λ=0.05). See our paper for full details.

---

## 📁 Project Structure

```
sevseg-yolo/
├── sevseg_yolo/                 # Core package
│   ├── model.py                 # SevSegYOLO: unified inference
│   ├── mask_generator_v3.py     # Bimodal channel selection (default)
│   ├── mask_generator_v2.py     # Variance channel selection (legacy)
│   ├── convert.py               # LabelMe → YOLO+Score converter
│   ├── evaluation.py            # Scoring metrics
│   ├── export.py                # ONNX export
│   ├── tensorrt_deploy.py       # TensorRT deployment
│   ├── visualization.py         # Plotting utilities
│   └── utils.py                 # Helpers
├── ultralytics/                 # Modified Ultralytics
│   ├── nn/modules/head.py       # ScoreHead + ScoreDetect
│   ├── utils/loss.py            # Gaussian NLL loss
│   ├── data/                    # 6-column label support
│   └── cfg/models/26/           # Model YAML configs
├── configs/train_score.yaml     # Training config template
├── tools/                       # CLI demos
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

## ⚠️ Important Notes

- **MixUp = 0**: mandatory for severity training
- **Severity range**: 0.0–10.0 (internally normalized to 0.0–1.0)
- **MaskGenerator**: approximate masks from features, not pixel-perfect segmentation
- **σ = 0.1**: Gaussian NLL fixed std, derived from ±1 annotation noise
- **No opencv-contrib**: only standard `opencv-python` needed

---

## 📖 Citation

```bibtex
@article{sevseg_yolo_2026,
  title={SevSeg-YOLO: A Unified Detection, Severity Scoring, and Annotation-Free
         Approximate Segmentation Framework for Industrial Defects},
  author={SevSeg-YOLO Contributors},
  year={2026}
}
```

## 📜 License

[AGPL-3.0](LICENSE). Modified Ultralytics code retains its original AGPL-3.0 license.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 foundation
- [LabelMe](https://github.com/wkentaro/labelme) — Annotation tooling
