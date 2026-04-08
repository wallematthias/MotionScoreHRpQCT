# MotionScoreHRpQCT

![MotionScoreHRpQCT logo](resources/MotionScoreHRpQCT.png)

Motion scoring for HR-pQCT scans using deep convolutional neural networks.

This refactor provides a dataset-first pipeline with BIDS-style derivatives and review-state persistence for direct Slicer integration.

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

Place torch model files (for example `DNN_0.pt` ... `DNN_9.pt`) in one of these locations:

- `./models`
- `motionscore/models`
- explicit `--model-dir <path>`

Model weights are licensed for usage tracking. In the current deployment configuration, licenses are automatically granted at signup.

## CLI Usage

### 1) Discover scans

```bash
motionscore discover /path/to/dataset
motionscore discover /path/to/dataset --json
```

### 2) Run prediction + initialize review tables

```bash
motionscore predict /path/to/dataset --confidence-threshold 75
# blinded operator training mode
motionscore predict /path/to/dataset --training-mode
# optional: restrict to one scan_id (repeat flag for multiple)
motionscore predict /path/to/dataset --scan-id sub-001_site-tibia_ses-T1_abcdef1234
# optional quick-look PNG controls
motionscore predict /path/to/dataset --preview-panels 5
motionscore predict /path/to/dataset --no-preview-png
```

Default output root:

```text
/path/to/dataset/derivatives/MotionScore
```

Optional custom output root:

```bash
motionscore predict /path/to/dataset --output-root /tmp/results
```

### 3) Update review threshold policy

```bash
motionscore review-init /path/to/dataset/derivatives/MotionScore --confidence-threshold 90
# keep/enable blinded operator training mode
motionscore review-init /path/to/dataset/derivatives/MotionScore --training-mode
```

`--confidence-threshold 100` effectively requires review of all scans.
When `--training-mode` is enabled, pending scans are always operator-first and AI prediction is revealed after manual grading.

### 4) Apply manual review decision for one scan

```bash
motionscore review-apply /path/to/dataset/derivatives/MotionScore \
  --scan-id sub-001_site-tibia_ses-T1_abcdef1234 \
  --manual-grade 3 \
  --reviewer mwalle
```

### 4b) Clear manual grades for re-review

```bash
# clear one operator across all scans
motionscore review-clear /path/to/dataset/derivatives/MotionScore --reviewer opA

# clear everyone
motionscore review-clear /path/to/dataset/derivatives/MotionScore --all-reviewers
```

### 5) Generate on-demand Grad-CAM attention map

```bash
motionscore explain /path/to/dataset/derivatives/MotionScore \
  --scan-id sub-001_site-tibia_ses-T1_abcdef1234
```

### 6) Export final grade table

```bash
motionscore export /path/to/dataset/derivatives/MotionScore
```

Writes `motion_grades.tsv` at the derivatives root (or custom `--output`).
Export includes current per-scan review state plus machine-readable multi-reviewer summary columns:
- `reviewer_count`
- `reviewers` (pipe-delimited reviewer IDs)
- `consensus_method` (currently `mean_manual_grade`)
- `consensus_mean_manual_grade`
- `consensus_grade_rounded`
- dynamic reviewer slots: `reviewer_1_id`, `reviewer_1_grade`, `reviewer_2_id`, `reviewer_2_grade`, ...

## Slicer Integration Contract

This repository is core logic only. A separate Slicer extension should:

1. run `motionscore predict ...` from a `Run` button,
2. run `motionscore review-apply ...` as reviewers step through scans,
3. request `motionscore explain ...` on demand to overlay Grad-CAM maps,
4. optionally show `preview/*_preview.png` for quick QC,
5. load all outputs from derivatives without ad hoc state files.

## Citation

If you use this software, please cite:

Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*, 166, p.116607. https://doi.org/10.1016/j.bone.2022.116607
