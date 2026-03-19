<div align="center">

# SevSeg-YOLO

**面向工业缺陷的统一检测、严重程度评分与零标注近似分割框架**

<br>

[![Python](https://img.shields.io/badge/python-≥3.8-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-≥1.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](CHANGELOG.md)

一个模型 &nbsp;·&nbsp; 三个任务 &nbsp;·&nbsp; 单次推理 &nbsp;·&nbsp; 无需掩膜标注

<br>

[中文](README_zh.md) &nbsp;|&nbsp; [English](README.md) &nbsp;|&nbsp; [贡献指南](CONTRIBUTING.md) &nbsp;|&nbsp; [更新日志](CHANGELOG.md)

</div>

<br>

## 为什么需要 SevSeg-YOLO？

设想一条精密光学镜片检测产线，每分钟扫描 **60+** 件产品。对每个缺陷，质量决策链需要回答三个问题：

| 问题 | 传统方案 | 痛点 |
|:---|:---|:---|
| **有没有缺陷？** | 标准 YOLO 检测器 | ✅ 已解决 |
| **缺陷多严重？**（等级 2 可接受，等级 8 要报废） | 额外分类模型 | 增加模型 + 增加延迟 |
| **缺陷覆盖多大？**（ISO/GB 合规要求） | 实例分割 | 10–20 倍标注成本 |

**SevSeg-YOLO 用一个模型、一次推理同时回答以上三个问题** — 检测头直接预测连续严重程度分数，训练无关的 MaskGenerator 从 FPN 特征中提取近似二值掩膜。

<br>

## 核心特性

**高斯 NLL 严重程度头** — 将检测员评分的 ±1 分主观偏差建模为观测噪声。相比 Smooth L1（5-seed 统计）：**MAE ↓ 21.2%**、**Spearman ρ ↑ 54.3%**。

**MaskGenerator（零标注分割）** — 纯 CPU 后处理模块，利用 FPN 特征的双峰通道选择 + Canny 引导上采样生成像素级掩膜。**100% 有效率**，中位耗时 **1.13 ms**。

**实时部署** — 全 5 规模 TRT FP16 端到端 **< 10 ms**（**> 100 FPS**）。Nano 规模纯推理 **534 FPS**，Engine 大小 **7.4 MB**。

<br>

## 快速开始

### 安装

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e .
```

> 只需标准 `opencv-python`，不需要 `opencv-contrib`。

可选依赖：

```bash
pip install -e ".[export]"      # ONNX 导出
pip install -e ".[tensorrt]"    # TensorRT 部署
```

### 3 行推理

```python
from sevseg_yolo import SevSegYOLO

model = SevSegYOLO("best.pt")
result = model.predict("image.jpg")

for det in result.detections:
    print(f"{det.class_name}: 严重程度={det.severity:.1f}, 掩膜填充率={det.fill_ratio:.3f}")
```

<br>

## 完整使用流程

### 第一步 · 准备数据集

<details>
<summary><b>1.1 &nbsp; 使用 LabelMe 标注</b></summary>
<br>

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

评分参考：

| 分数 | 含义 | 处置 |
|:---:|:---|:---|
| 0 | 无缺陷 | — |
| 1 – 3 | 轻微 | 接受 |
| 4 – 6 | 中等 | 返工 |
| 7 – 10 | 严重 | 报废 |

</details>

<details>
<summary><b>1.2 &nbsp; 整理目录结构</b></summary>
<br>

```
my_dataset/
├── images/
│   ├── img_001.jpg
│   └── ...
└── jsons/
    ├── img_001.json          ← LabelMe 标注文件
    └── ...
```

</details>

<details>
<summary><b>1.3 &nbsp; 格式转换</b></summary>
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

输出：标准 YOLO 布局 + 6 列标签（`类别ID 中心x 中心y 宽度 高度 严重程度`）。

</details>

### 第二步 · 训练

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26m-score.yaml")
model.train(
    task="score_detect",
    data="my_dataset_yolo/data.yaml",
    pretrained="yolo26m.pt",
    epochs=105, batch=32, imgsz=640,
    mixup=0.0,   # ⚠️ 必须为 0 — severity 不能在混合图像间线性插值
)
```

> **建议：** 先用 `n`（nano）规模快速验证 → 再换 `m` 或 `l` → 关注 `score_loss` 下降曲线。

### 第三步 · 推理

<details>
<summary><b>Python API</b>（推荐）</summary>
<br>

```python
from sevseg_yolo import SevSegYOLO

model = SevSegYOLO("runs/score_detect/train/weights/best.pt")
result = model.predict("test.jpg")

for det in result.detections:
    print(f"类别: {det.class_name}, 严重程度: {det.severity:.1f}/10")
    print(f"  检测框: {det.bbox}, Mask: {det.mask.shape}, 填充率: {det.fill_ratio:.3f}")

# 可视化并保存
result.visualize().save("output.jpg")

# 按严重程度/置信度过滤
severe = result.filter(min_severity=7.0, min_confidence=0.5)

# 批量推理
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])

# JSON 序列化输出
data = result.to_dict()
```

</details>

<details>
<summary><b>命令行</b></summary>
<br>

```bash
python tools/predict_demo.py --weights best.pt --source test_images/ --save-dir outputs/
```

</details>

### 第四步 · 导出部署

<details>
<summary><b>ONNX + TensorRT</b></summary>
<br>

```python
from sevseg_yolo.export import export_scoreyolo_onnx
from sevseg_yolo.tensorrt_deploy import deploy_scoreyolo

# 导出 ONNX（支持可选 PCA 特征压缩）
export_scoreyolo_onnx(model.model, "model.onnx", imgsz=640, opset=17)

# 构建 TensorRT 引擎
deploy_scoreyolo("model.onnx", "model.engine", fp16=True, max_batch=4)
```

ONNX 输出格式：`det_output (B, K, 7)` → `[x1, y1, x2, y2, conf, cls, severity]`，可选 `feat_p3/p4/p5` 节点供 MaskGenerator 使用。

</details>

### 第五步 · 评估

```python
from sevseg_yolo.evaluation import full_score_evaluation, print_evaluation_report

metrics = full_score_evaluation(pred_scores, gt_scores)
print_evaluation_report(metrics)
# → MAE, Spearman ρ, ±1 容忍准确率, 低端/高端误判率,
#   分段 MAE, 11×11 混淆矩阵
```

<br>

## MaskGenerator 工作原理

**纯后处理模块** — 不参与训练，在 CPU 上运行。将 FPN 特征图中的缺陷隐式语义知识转化为显式二值掩膜，包含 6 个步骤：

1. **尺度自适应特征选择** — 根据 bbox 大小选择 P3/P4/P5
2. **双峰 Top-K 通道选择** — 挑选缺陷-正常分离度最高的通道
3. **多尺度加权融合** — 将选中通道合成为单一激活图
4. **Canny 边缘引导上采样** — 利用原图边缘信息引导激活图上采样
5. **自适应二值化** — 局部阈值化生成二值掩膜
6. **形态学后处理** — 闭运算 + 开运算去噪填补

**为什么用双峰选择（V3）？** 按方差选通道（V2）可能选到"背景纹理变化大"的通道。双峰选择测量 bbox 内最暗 30% 和最亮 30% 像素的均值差 — 直接度量"缺陷 vs 正常"的分离程度。

```python
model = SevSegYOLO("best.pt", mask_version="v3")  # 双峰选择（默认）
model = SevSegYOLO("best.pt", mask_version="v2")  # 方差选择（旧版）
model = SevSegYOLO("best.pt", mask_enabled=False)  # 仅检测 + 评分
```

<br>

## 模型库

全部结果为 5-seed 平均值，使用 Gaussian NLL（σ = 0.1, λ = 0.05）：

| 规模 | 参数量 | mAP@50 | Score MAE ↓ | Spearman ρ ↑ |
|:---:|:---:|:---:|:---:|:---:|
| **n**（nano） | 2.57 M | 0.513 | 1.317 | 0.742 |
| **s**（small） | 10.19 M | 0.573 | 1.306 | 0.720 |
| **m**（medium） | 22.19 M | 0.608 | 1.316 | 0.715 |
| **l**（large） | 26.59 M | 0.626 | 1.297 | 0.709 |
| **x**（xlarge） | 56.08 M | 0.623 | 1.224 | 0.744 |

模型配置文件位于 `ultralytics/cfg/models/26/yolo26{n,s,m,l,x}-score.yaml`。

<br>

## 项目结构

```
sevseg-yolo/
├── sevseg_yolo/                  # 核心包
│   ├── model.py                  # SevSegYOLO — 统一推理入口
│   ├── mask_generator_v3.py      # V3: 双峰通道选择（默认）
│   ├── mask_generator_v2.py      # V2: 方差通道选择（旧版兼容）
│   ├── convert.py                # LabelMe JSON → 6 列 YOLO 格式转换
│   ├── evaluation.py             # 严重程度评估指标 (MAE, Spearman ρ, 容忍准确率等)
│   ├── export.py                 # ONNX 导出 (可选 PCA 特征压缩)
│   ├── tensorrt_deploy.py        # TensorRT FP16/INT8 部署流水线
│   ├── visualization.py          # 散点图、排序曲线、混淆矩阵热力图
│   └── utils.py                  # 特征 Hook、坐标转换工具
│
├── ultralytics/                  # 修改的 Ultralytics (YOLO26 + ScoreDetect)
│   ├── nn/modules/head.py        # ScoreHead & ScoreDetect 检测头
│   ├── utils/loss.py             # 高斯 NLL 评分损失
│   └── cfg/models/26/            # yolo26{n,s,m,l,x}-score.yaml
│
├── configs/                      # 训练配置模板
├── tools/                        # 命令行脚本 (predict_demo, visualize_masks)
└── pyproject.toml
```

<br>

## 注意事项

| 条目 | 说明 |
|:---|:---|
| **MixUp = 0** | 必须关闭 — severity 不能在混合图像间线性插值 |
| **Severity 范围** | 0.0 – 10.0（内部归一化到 0 – 1） |
| **MaskGenerator** | 基于特征的近似掩膜，非像素级精确 |
| **Gaussian σ = 0.1** | 源自 ±1 标注员间主观噪声 |
| **OpenCV** | 仅需标准 `opencv-python`，**不需要** `opencv-contrib` |

<br>

## 引用

```bibtex
@software{sevseg_yolo_2026,
  title  = {SevSeg-YOLO: Unified Detection, Severity Scoring, and
            Annotation-Free Approximate Segmentation for Industrial Defects},
  author = {SevSeg-YOLO Contributors},
  year   = {2026},
  url    = {https://github.com/sevseg-yolo/sevseg-yolo}
}
```

## 许可证

[AGPL-3.0](LICENSE)。修改的 Ultralytics 代码保留其原始 AGPL-3.0 许可证。

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 框架
- [LabelMe](https://github.com/wkentaro/labelme) — 标注工具
