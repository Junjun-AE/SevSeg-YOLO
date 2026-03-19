# Contributing to SevSeg-YOLO

Thank you for your interest in contributing! This document provides guidelines for contributing to SevSeg-YOLO.

## Development Setup

```bash
git clone https://github.com/sevseg-yolo/sevseg-yolo.git
cd sevseg-yolo
pip install -e ".[export]"
```

## Project Structure

```
sevseg_yolo/          # Core Python package
├── model.py          # SevSegYOLO: unified inference entry point
├── mask_generator_v3.py  # V3: bimodal channel selection (default)
├── mask_generator_v2.py  # V2: variance channel selection (legacy)
├── convert.py        # LabelMe JSON → 6-column YOLO format
├── evaluation.py     # Severity scoring metrics
├── export.py         # ONNX export
├── tensorrt_deploy.py    # TensorRT deployment
├── visualization.py  # Plotting utilities
└── utils.py          # Feature hooks, coordinate helpers

ultralytics/          # Modified Ultralytics (YOLO26 + ScoreDetect)
├── nn/modules/head.py    # ScoreHead, ScoreDetect
├── utils/loss.py         # GaussianNLL score loss
├── data/                 # 6-column label parsing, augmentation
└── cfg/models/26/        # yolo26{n,s,m,l,x}-score.yaml
```

## Code Style

- Python 3.8+ compatible
- Type hints where practical
- Docstrings for all public methods

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with clear commit messages
4. Ensure existing tests still pass
5. Submit a pull request with a description of the changes

## Reporting Issues

When reporting a bug, please include:
- Python and PyTorch versions
- GPU model and CUDA version (if applicable)
- Minimal code to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under AGPL-3.0.
