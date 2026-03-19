<div align="center">

<img src="https://img.shields.io/badge/🔬-SevSeg--YOLO-blue?style=for-the-badge&labelColor=0a0a23" alt="SevSeg-YOLO" height="40">

# SevSeg-YOLO

### 面向工业缺陷的统一检测、严重程度评分与零标注近似分割框架

**一个模型 · 三个任务 · 单次前向传播 · 无需掩膜标注**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-AGPL_3.0-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange?style=flat-square)](CHANGELOG.md)

**中文** · **[📖 English](README.md)**

---

SevSeg-YOLO 在 YOLO26 基础上增加轻量级 **ScoreHead** 分支（参数增量 <3%），在**单次前向传播**中同时输出**缺陷检测框**、**\[0,10\] 连续严重程度分数**和**近似轮廓掩膜**——**无需任何掩膜标注**。

</div>

---

## ✨ 为什么选择 SevSeg-YOLO？

设想一条精密光学镜片检测产线，每分钟扫描 60+ 件产品。对每个缺陷，质量决策链需要回答三个问题：

| 问题 | 传统方案 | SevSeg-YOLO |
|:---|:---|:---|
| **是否有缺陷？** | YOLO 检测器 ✅ | ✅ 内置检测 |
| **缺陷多严重？**（等级2可接受，等级8需报废） | 额外分类模型（更多延迟） | ✅ ScoreHead 输出 0–10 连续分数 |
| **缺陷覆盖多大面积？**（ISO/GB 合规） | 像素级分割（标注成本是检测的 10–20 倍） | ✅ MaskGenerator 从特征图推导掩膜，**零额外标注** |

SevSeg-YOLO 用**一个模型、一次推理**同时回答以上三个问题。无需级联多模型，无需额外标注，无需额外延迟。

### 核心创新

- 🧪 **高斯 NLL 损失** — 将检测员评分的 ±1 分主观偏差建模为观测噪声，MAE 比 Smooth L1 降低 21.2%
- 🎭 **MaskGenerator** — 纯后处理模块，利用 FPN 特征的双峰通道选择 + Canny 引导上采样生成像素级掩膜。CPU 中位耗时仅 1.13ms，100% 有效率
- ⚡ **实时部署** — 全 5 规模 TRT FP16 端到端 < 10ms（> 100 FPS）

---

## 🏗️ 架构

```
  输入图像 (640×640)
         │
         ▼
  ┌─────────────────────────┐
  │  YOLO26 骨干网络 + 特征  │
  │  金字塔 (FPN/PAN)       │
  │                         │
  │   P3(s=8) P4(s=16) P5  │
  └──────────┬──────────────┘
             │
  ┌──────────▼──────────────┐
  │    ScoreDetect 检测头    │
  │                         │
  │  ┌─────┐ ┌─────┐ ┌────────────┐
  │  │检测头│ │分类头│ │ ScoreHead  │ ← 新增 (<3% 参数)
  │  │(坐标)│ │(类别)│ │ DWConv→1×1 │
  │  │     │ │     │ │ →Sigmoid×10│
  │  └──┬──┘ └──┬──┘ └─────┬──────┘
  └─────┼───────┼──────────┼────────┘
        └───┬───┘          │
            ▼              ▼
       检测结果          严重程度
  (x1,y1,x2,y2,       (0 – 10)
   置信度, 类别)
            │
            ▼
  ┌─────────────────────────┐
  │    MaskGenerator        │  (后处理, CPU, 不参与训练)
  │                         │
  │  1. 尺度自适应特征选择    │
  │  2. 双峰 Top-K 通道选择  │
  │  3. 多尺度加权融合       │
  │  4. Canny 边缘引导上采样 │
  │  5. 自适应二值化         │
  │  6. 形态学后处理         │
  └──────────┬──────────────┘
             ▼
          最终输出:
  检测框 + 类别 + 置信度
  + 严重程度 + 二值掩膜
```

---

## 🚀 安装

### 从源码安装（推荐）

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e .
```

### 依赖

SevSeg-YOLO 只需要标准包——**不需要 `opencv-contrib`**：

```
torch >= 1.8.0
opencv-python >= 4.6.0
numpy, scipy, matplotlib, pillow, pyyaml
```

如需导出 ONNX：`pip install -e ".[export]"`

如需 TensorRT：`pip install -e ".[tensorrt]"`

---

## 📖 完整使用教程

### 第一步：准备数据集

#### 1.1 使用 LabelMe 标注

安装 [LabelMe](https://github.com/wkentaro/labelme)，用矩形框标注缺陷。在 `description` 字段写入严重程度分数：

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

> 💡 **评分参考**：0 = 无缺陷，1–3 = 轻微（可接受），4–6 = 中等（需返工），7–10 = 严重（报废）。请根据实际质量标准调整。

#### 1.2 整理目录结构

```
my_dataset/
├── images/          ← 原始图片
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── jsons/           ← LabelMe 标注文件
    ├── img_001.json
    ├── img_002.json
    └── ...
```

#### 1.3 格式转换

```python
from sevseg_yolo.convert import convert_dataset

convert_dataset(
    images_dir="my_dataset/images",
    jsons_dir="my_dataset/jsons",
    output_dir="my_dataset_yolo",
    val_ratio=0.2,   # 80% 训练集, 20% 验证集
)
```

转换后的目录结构：

```
my_dataset_yolo/
├── images/
│   ├── train/       ← 训练图片
│   └── val/         ← 验证图片
├── labels/
│   ├── train/
│   │   └── img_001.txt   ← "0 0.500 0.300 0.120 0.080 7.5"
│   └── val/
└── data.yaml              ← 自动生成的配置文件
```

**标签格式**——标准 YOLO 格式 + 第 6 列严重程度分数：

```
# 类别ID  中心x  中心y  宽度  高度  严重程度
0  0.500  0.300  0.120  0.080  7.5
1  0.250  0.600  0.050  0.040  3.0
```

### 第二步：训练模型

```python
from ultralytics import YOLO

# 选择模型规模: n(最快) / s / m(推荐) / l / x(最精确)
model = YOLO("ultralytics/cfg/models/26/yolo26m-score.yaml")

model.train(
    task="score_detect",
    data="my_dataset_yolo/data.yaml",
    pretrained="yolo26m.pt",       # YOLO26 预训练骨干权重
    epochs=105,
    batch=32,
    imgsz=640,
    mixup=0.0,                     # ⚠️ 必须为 0
)
```

> ⚠️ **MixUp 必须关闭**——严重程度分数不能在混合图像间线性插值。"严重裂纹×0.3 + 轻微划痕×0.7 = 严重程度5.2" 没有物理意义。

**训练建议：**
- 先用 `n`（nano）规模快速验证流程，再换大模型
- 使用项目提供的 `configs/train_score.yaml` 作为起点
- ScoreHead 从零开始训练，骨干网络加载预训练权重
- 训练过程中关注 `score_loss`，它应该和 `box_loss`、`cls_loss` 一起稳步下降

### 第三步：推理

#### Python API（推荐）

```python
from sevseg_yolo import SevSegYOLO

# 加载模型
model = SevSegYOLO("runs/score_detect/train/weights/best.pt")

# 单张推理
result = model.predict("test_image.jpg")

for det in result.detections:
    print(f"类别: {det.class_name}")
    print(f"  严重程度: {det.severity:.1f} / 10")
    print(f"  置信度: {det.confidence:.2f}")
    print(f"  检测框: {det.bbox}")
    print(f"  Mask 形状: {det.mask.shape}")
    print(f"  Mask 填充率: {det.fill_ratio:.3f}")

# 批量推理
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])
```

#### 命令行

```bash
python tools/predict_demo.py \
    --weights best.pt \
    --source test_images/ \
    --save-dir outputs/ \
    --conf 0.15
```

#### 可视化保存

```python
# 保存带检测框、严重程度标签、掩膜叠加的可视化图
result.visualize().save("output.jpg")
```

### 第四步：导出部署

#### ONNX 导出

```python
from sevseg_yolo.export import export_scoreyolo_onnx

export_scoreyolo_onnx(
    model.model,
    "model.onnx",
    imgsz=640,
    opset=17,
)
```

#### TensorRT 部署

```python
from sevseg_yolo.tensorrt_deploy import deploy_scoreyolo

deploy_scoreyolo(
    "model.onnx",
    "model.engine",
    fp16=True,          # FP16 量化
    max_batch=4,
)
```

### 第五步：评估严重程度评分

```python
from sevseg_yolo.evaluation import full_score_evaluation

metrics = full_score_evaluation(pred_scores, gt_scores)
print(f"MAE: {metrics['mae']:.3f}")
print(f"Spearman ρ: {metrics['spearman_rho']:.3f}")
print(f"±1 容忍准确率: {metrics['tolerance_acc']:.1%}")
```

---

## 🎭 MaskGenerator 工作原理

MaskGenerator 是**纯后处理模块**——不参与训练，在 CPU 上运行。它利用检测模型已经产生的 FPN 特征图（P3/P4/P5）来推导像素级缺陷掩膜。

**核心思路**：经过缺陷检测训练的骨干网络已经隐式学会了区分缺陷和正常区域——缺陷区域在特征通道上呈现独特的激活模式。MaskGenerator 将这种隐式知识转化为显式的二值掩膜。

| 步骤 | 操作 | 说明 |
|:---:|:---|:---|
| 1 | **尺度自适应特征选择** | 根据模型规模选择最佳特征组合（n/s: Layer2+P3+P4; m/l/x: P3+P4+P5） |
| 2 | **双峰 Top-K 通道选择** | 在 bbox 区域内，计算每个通道最暗 30% 与最亮 30% 像素的均值差。选分离度最大的 K 个通道——这些通道最能区分缺陷和正常区域 |
| 3 | **多尺度加权融合** | 各尺度激活图上采样到最高分辨率（P3 级别），按权重加权平均 |
| 4 | **Canny 边缘引导上采样** | 用原图的边缘信息引导低分辨率激活图的上采样——边缘处保持锐利，平滑处去除噪声 |
| 5 | **自适应二值化** | `cv2.adaptiveThreshold` 将连续激活值转换为 {0, 1} 二值掩膜 |
| 6 | **形态学后处理** | 闭操作（填充小孔洞）+ 开操作（去除噪声小点）→ 最终二值掩膜 |

---

## 📊 模型库

| 规模 | 参数 | mAP@50 | MAE | Spearman ρ |
|:---:|:---:|:---:|:---:|:---:|
| **n** | 2.57M | 0.513 | 1.317 | 0.742 |
| **s** | 10.19M | 0.573 | 1.306 | 0.720 |
| **m** | 22.19M | 0.608 | 1.316 | 0.715 |
| **l** | 26.59M | 0.626 | 1.297 | 0.709 |
| **x** | 56.08M | 0.623 | 1.224 | 0.744 |

> 以上为 5-seed 统计实验结果（Gaussian NLL σ=0.1, λ=0.05）。完整实验数据请参阅论文。

---

## 📁 项目结构

```
sevseg-yolo/
├── sevseg_yolo/                 # 核心 Python 包
│   ├── model.py                 # SevSegYOLO 统一推理入口
│   ├── mask_generator_v3.py     # 双峰通道选择（默认）
│   ├── mask_generator_v2.py     # 方差通道选择（旧版兼容）
│   ├── convert.py               # LabelMe → YOLO+Score 格式转换
│   ├── evaluation.py            # 评估指标（MAE, Spearman ρ, 容忍准确率）
│   ├── export.py                # ONNX 导出
│   ├── tensorrt_deploy.py       # TensorRT 部署
│   ├── visualization.py         # 可视化工具
│   └── utils.py                 # 辅助函数
├── ultralytics/                 # 修改的 Ultralytics
│   ├── nn/modules/head.py       # ScoreHead + ScoreDetect
│   ├── utils/loss.py            # 高斯 NLL 损失
│   ├── data/                    # 6 列标签支持
│   └── cfg/models/26/           # 模型 YAML 配置
├── configs/train_score.yaml     # 训练配置模板
├── tools/                       # 命令行演示脚本
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

## ⚠️ 注意事项

- **MixUp = 0**：严重程度训练必须关闭 MixUp
- **Severity 范围**：0.0–10.0（训练时内部归一化到 0.0–1.0）
- **MaskGenerator**：从特征图推导的近似掩膜，不是像素级精确分割
- **σ = 0.1**：高斯 NLL 固定标准差，由工业标注噪声（±1 分）推导
- **无需 opencv-contrib**：只需标准 `opencv-python`

---

## 📖 引用

```bibtex
@article{sevseg_yolo_2026,
  title={SevSeg-YOLO: A Unified Detection, Severity Scoring, and Annotation-Free
         Approximate Segmentation Framework for Industrial Defects},
  author={SevSeg-YOLO Contributors},
  year={2026}
}
```

## 📜 许可证

[AGPL-3.0](LICENSE)。修改的 Ultralytics 代码保留其原始 AGPL-3.0 许可证。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 基础框架
- [LabelMe](https://github.com/wkentaro/labelme) — 标注工具
