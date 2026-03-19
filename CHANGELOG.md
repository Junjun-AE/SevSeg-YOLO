# Changelog

All notable changes to SevSeg-YOLO will be documented in this file.

## [2.0.0] - 2026-03-19

### Added
- **MaskGeneratorV3**: Bimodal channel selection for improved defect/normal separation
  - Selects feature channels by measuring the gap between top-30% and bottom-30% pixel values within each bbox
  - Channels with the largest bimodal gap are most discriminative for defect boundaries
  - All other components (multi-scale fusion, Canny-guided upsampling, adaptive binarization, morphology) remain identical to V2
- V3 is now the default `mask_version` in `SevSegYOLO`
- `CHANGELOG.md`, `CONTRIBUTING.md` added for open-source release

### Changed
- Default `mask_version` changed from `"v2"` to `"v3"`
- `opencv-python` replaces `opencv-contrib-python` in dependencies (no contrib modules needed)
- Removed unused `polars` dependency
- Version bumped to 2.0.0

### Fixed
- Binary mask visualization no longer draws bbox rectangles when mask is empty (`app.py`)

## [1.0.0] - 2026-03-01

### Added
- Initial release
- SevSegYOLO unified inference class
- ScoreDetect head (Gaussian NLL loss, severity 0-10)
- MaskGeneratorV2 (Top-K variance + Canny edge-guided upsampling)
- LabelMe → YOLO+Score format converter
- ONNX export with optional PCA feature compression
- TensorRT FP16/INT8 deployment pipeline
- Evaluation metrics: MAE, Spearman ρ, tolerance accuracy
- 5 model scales: nano / small / medium / large / xlarge
