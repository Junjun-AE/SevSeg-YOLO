<div align="center">

# SevSeg-YOLO

**Unified Detection, Severity Scoring, and Annotation-Free Approximate Segmentation for Industrial Defects**

One model, three tasks: defect detection + severity scoring (0–10) + zero-annotation mask generation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange)](CHANGELOG.md)

</div>

---

## What is SevSeg-YOLO?

SevSeg-YOLO extends [YOLO26](https://github.com/ultralytics/ultralytics) with two additions:

1. **Score Head** — a lightweight branch that predicts defect severity (0–10) alongside standard detection, trained with Gaussian NLL loss.
2. **MaskGenerator** — a post-processing module that produces pixel-level defect masks from detection features alone, requiring **zero segmentation annotations**.

| | Traditional Approach | SevSeg-YOLO |
|---|---|---|
| Models needed | 3 (Detector + Classifier + Segmentor) | **1 unified model** |
| Annotations needed | Boxes + Classes + Pixel masks | **Boxes + Severity score** |
| Inference passes | 3 sequential | **1 forward pass** |

## Key Features

- **Three-in-one inference**: detection + severity scoring + approximate segmentation from a single model
- **Zero-annotation masks**: MaskGeneratorV3 derives pixel-level masks from P3/P4/P5 feature activations, no mask labels required
- **Bimodal channel selection (V3)**: automatically selects feature channels that best separate defect from normal regions within each bbox
- **YOLO26-Score architecture**: NMS-free end-to-end detection with only ~0.1M additional parameters for the Score Head
- **Gaussian NLL loss**: severity prediction with learned uncertainty
- **Full deployment pipeline**: PyTorch → ONNX → TensorRT FP16/INT8
- **5 model scales**: nano (2.5M) / small (10M) / medium (22M) / large (27M) / xlarge (56M)

## Quick Start

### Installation

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e .
```

### Inference (3 lines)

```python
from sevseg_yolo import SevSegYOLO

model = SevSegYOLO("best.pt")
result = model.predict("image.jpg")

for det in result.detections:
    print(f"{det.class_name}: severity={det.severity:.1f}, conf={det.confidence:.2f}")
    print(f"  bbox={det.bbox}, mask_fill={det.fill_ratio:.3f}")
```

### Data Preparation

Annotate with [LabelMe](https://github.com/wkentaro/labelme). Add severity in the `description` field:

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

Convert to 6-column YOLO format:

```python
from sevseg_yolo.convert import convert_dataset

convert_dataset(
    images_dir="dataset/images",
    jsons_dir="dataset/jsons",
    output_dir="dataset_yolo",
    val_ratio=0.2,
)
```

### Training

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26m-score.yaml")
model.train(
    task="score_detect",
    data="dataset_yolo/data.yaml",
    pretrained="yolo26m.pt",
    epochs=105,
    batch=32,
    imgsz=640,
    mixup=0.0,  # MUST be 0 for severity training
)
```

### Export & Deploy

```python
from sevseg_yolo.export import export_scoreyolo_onnx
from sevseg_yolo.tensorrt_deploy import deploy_scoreyolo

# PyTorch → ONNX
export_scoreyolo_onnx(model.model, "model.onnx", imgsz=640)

# ONNX → TensorRT FP16
deploy_scoreyolo("model.onnx", "model.engine", fp16=True)
```

## Architecture

```
Input Image
      │
      ▼
┌─────────────────┐
│  YOLO26 Backbone │  (CSP-DarkNet)
│  + FPN/PAN Neck  │  → P3 (stride 8), P4 (stride 16), P5 (stride 32)
└────────┬────────┘
         │
    ┌────┼────┬──────────┐
    ▼    ▼    ▼          ▼
  Box   Cls  DFL    Score Head (NEW)
  Head  Head Head   DWConv→1×1→Sigmoid×10
    │    │    │          │
    └────┼────┘          │
         ▼               ▼
   Detection Output   Severity (0–10)
   (x1,y1,x2,y2,conf,cls)
         │
         ▼
  ┌──────────────────┐
  │  MaskGeneratorV3  │  (post-processing, no training required)
  │                   │
  │  Per bbox:        │
  │  1. Crop P3/P4/P5 at bbox region          │
  │  2. Bimodal Top-K channel selection        │
  │  3. L2 activation + multi-scale fusion     │
  │  4. Canny edge-guided upsampling           │
  │  5. Adaptive binarization + morphology     │
  └──────────┬───────┘
             ▼
  Final Output: boxes + classes + conf + severity + binary masks
```

### MaskGeneratorV3 vs V2

| | V2 (legacy) | V3 (default) |
|---|---|---|
| Channel selection | Top-K by spatial **variance** | Top-K by **bimodal gap** |
| Channel combination | L2 norm (equal weight) | L2 norm (equal weight) |
| Upsampling | Canny edge-guided | Canny edge-guided |
| Binarization | Adaptive threshold | Adaptive threshold |
| Morphology | Close + Open | Close + Open |

The only difference: V3 selects channels by measuring the separation between the brightest 30% and darkest 30% of pixels within each bbox crop. This picks channels that best distinguish defect from normal regions, rather than channels with the highest overall variance (which may just capture background texture).

## Label Format

Standard YOLO format extended with a 6th column for severity:

```
# class_id  center_x  center_y  width  height  severity
0  0.500  0.300  0.120  0.080  7.5
1  0.250  0.600  0.050  0.040  3.0
```

- Severity range: 0.0 (no defect) to 10.0 (most severe)
- Use `-1` for unlabeled samples (treated as NaN during training)

## Model Zoo

| Model | Params | mAP50 | Score MAE | Spearman ρ |
|-------|--------|-------|-----------|-----------|
| YOLO26n-Score | 2.57M | 51.3% | — | — |
| YOLO26s-Score | 10.19M | 57.3% | — | — |
| **YOLO26m-Score** | **22.19M** | **60.8%** | — | — |
| YOLO26l-Score | 26.59M | 62.6% | — | — |
| YOLO26x-Score | 56.08M | 62.3% | — | — |

> Score MAE and Spearman ρ depend on dataset. Train on your own data for severity metrics.

## Evaluation Metrics

```python
from sevseg_yolo.evaluation import full_score_evaluation

metrics = full_score_evaluation(pred_scores, gt_scores)
# Returns: MAE, Spearman ρ, ±1 tolerance accuracy,
#          low/high-end misjudge rate, per-class MAE
```

## Project Structure

```
sevseg-yolo/
├── sevseg_yolo/                 # Core Python package
│   ├── model.py                 # SevSegYOLO unified inference class
│   ├── mask_generator_v3.py     # V3: bimodal channel selection (default)
│   ├── mask_generator_v2.py     # V2: variance channel selection (legacy)
│   ├── convert.py               # LabelMe JSON → 6-column YOLO format
│   ├── evaluation.py            # Severity scoring metrics (MAE, Spearman, etc.)
│   ├── export.py                # ONNX export with optional PCA compression
│   ├── tensorrt_deploy.py       # TensorRT FP16/INT8 engine builder
│   ├── visualization.py         # Scatter plots, confusion matrices, training curves
│   ├── predict.py               # CLI prediction wrapper
│   └── utils.py                 # Feature hooks, coordinate mapping utilities
├── ultralytics/                 # Modified Ultralytics (7-layer changes)
│   ├── nn/modules/head.py       # ScoreHead + ScoreDetect
│   ├── utils/loss.py            # GaussianNLL score loss
│   ├── utils/nms.py             # 7-column NMS support
│   ├── data/dataset.py          # 6-column label loading
│   ├── data/augment.py          # Score-aware augmentation
│   ├── models/yolo/model.py     # score_detect task registration
│   └── cfg/models/26/           # yolo26{n,s,m,l,x}-score.yaml
├── configs/
│   └── train_score.yaml         # Recommended training config
├── tools/
│   ├── predict_demo.py          # CLI inference demo
│   └── visualize_masks.py       # Mask visualization tool
├── pyproject.toml
├── LICENSE                      # AGPL-3.0
├── CHANGELOG.md
└── CONTRIBUTING.md
```

## Important Notes

- **MixUp must be 0**: severity scores cannot be meaningfully interpolated between mixed images
- **Severity range**: 0.0–10.0 (internally normalized to 0.0–1.0 during training)
- **MaskGenerator is post-processing**: masks are approximate and derived from detection features, not pixel-perfect segmentation
- **No opencv-contrib needed**: all image processing uses standard `opencv-python`

## Citation

If you use SevSeg-YOLO in your research, please cite:

```bibtex
@software{sevseg_yolo_2026,
  title={SevSeg-YOLO: Unified Detection, Severity Scoring, and Annotation-Free Approximate Segmentation},
  author={SevSeg-YOLO Contributors},
  year={2026},
  url={https://github.com/sevseg-yolo/sevseg-yolo}
}
```

## License

This project is licensed under [AGPL-3.0](LICENSE). The modified Ultralytics code retains its original AGPL-3.0 license.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO26 foundation
- [LabelMe](https://github.com/wkentaro/labelme) for annotation tooling
