<img src="resources/MotionScoreHRpQCT.png" alt="MotionScoreHRpQCT logo" width="240" />

# MotionScoreHRpQCT

[![CI](https://github.com/wallematthias/MotionScoreHRpQCT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wallematthias/MotionScoreHRpQCT/actions/workflows/ci.yml)
[![Coverage Gate](https://img.shields.io/badge/coverage%20gate-70%25-brightgreen)](https://github.com/wallematthias/MotionScoreHRpQCT/blob/main/.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/motionscorehrpqct.svg)](https://pypi.org/project/motionscorehrpqct/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/motionscorehrpqct.svg)](https://pypi.org/project/motionscorehrpqct/)

Motion scoring for HR-pQCT scans using deep convolutional neural networks.

This refactor provides a dataset-first pipeline with BIDS-style derivatives and review-state persistence for direct Slicer integration.

Related repositories:
- Core pipeline (this repo): https://github.com/wallematthias/MotionScoreHRpQCT
- Slicer extension: https://github.com/wallematthias/SlicerMotionScoreHRpQCT

## What Changed In v2

- Legacy CLI commands `grade` and `confirm` are removed.
- New dataset-driven commands: `discover`, `predict`, `review-init`, `review-apply`, `explain`, `export`.
- Default output structure is now:

```text
<dataset_root>/derivatives/MotionScore/
  index.tsv
  dataset_description.json
  <mirrored-source-path-or-flat-aim-name>/
    predictions/predictions.tsv
    preview/<scan_id>_preview.png
    preview/<scan_id>_slice_profile.png
    review/review.tsv
    review/review.json
    review/review_audit.tsv
    explain/<scan_id>_gradcam.mha
```

- AIM reading now uses `aimio-py`.
- Python baseline is now `>=3.10`.
- Output path mapping:
  - Flat input (`*.AIM` directly in dataset root): outputs are grouped under folder named after each AIM file stem.
  - Structured input (nested folders): outputs mirror the source folder structure under `MotionScore`.
- Raw-vs-mask identification:
  - Primary: AIM header processing log (ISQ-origin markers indicate raw images).
  - Fallback: filename-based heuristics when header signal is unavailable.

## Installation

```bash
conda create -n motionscore python=3.10 -y
conda activate motionscore

# Clone
# git clone <repo-url>
# cd MotionScoreHRpQCT

# Install CLI + torch inference backend
pip install -e ".[torch]"
```

## Models

Use a model registry rooted at `--model-root` (default `~/.motionscore/MotionScore/models`).

Each registered profile points to a directory containing torch checkpoints:
- `DNN_0.pt`, `DNN_1.pt`, ... (ensemble members)
- `model_registry.json` at the model root

Model weights are licensed for usage tracking. In the current deployment configuration, licenses are automatically granted at signup.

## CLI Usage

For most users, day-to-day grading and review should be done in the Slicer app.
In this core CLI, the two most useful workflows are batch prediction and retraining.

### 1) Predict a folder of scans

```bash
motionscore predict /path/to/dataset --model-id base-v1
```

Default output root:
- `/path/to/dataset/derivatives/MotionScore`

Review confidence policy is configured separately with `motionscore review-init --confidence-threshold ...`.

### 2) Retrain from reviewed data

```bash
motionscore train-prepare /path/to/dataset/derivatives/MotionScore \
  --output /path/to/dataset/derivatives/MotionScore/training/train_manifest.tsv \
  --slice-count 8 \
  --seed 13 \
  --cv-folds 10 \
  --min-auto-confidence 0.70 \
  --include-auto-without-manual

motionscore train \
  --manifest /path/to/dataset/derivatives/MotionScore/training/train_manifest.tsv \
  --model-root ~/.motionscore/MotionScore/models \
  --init-model-id base-v1 \
  --early-stopping-patience 10 \
  --seed 13 \
  --output-model-dir ~/.motionscore/MotionScore/models/knee-v1
```

Training writes:
- `training_metrics.json`
- `training_plot_live.png` (updated every epoch)
- `training_plot.png` (final summary plot)
- `training_plot_model_<n>.png` (per-ensemble-model curves)

### CLI Reference

For all advanced/headless commands (`discover`, `review-*`, `export`, `explain`, `model-*`), see:
- [CLI_REFERENCE.md](CLI_REFERENCE.md)

## Use With 3D Slicer

For day-to-day grading and retraining, use the Slicer app:
- https://github.com/wallematthias/SlicerMotionScoreHRpQCT

This repository provides the core CLI/pipeline used by that extension.

## Citation

If you use this software, please cite:

Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*, 166, p.116607. https://doi.org/10.1016/j.bone.2022.116607
