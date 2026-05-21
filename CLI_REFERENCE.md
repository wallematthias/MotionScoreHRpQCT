# MotionScoreHRpQCT CLI Reference

This document lists advanced/headless commands that are available in the core CLI.
For most interactive workflows, use the Slicer app:
- https://github.com/wallematthias/SlicerMotionScoreHRpQCT

## Main Commands

- `motionscore discover <dataset_root>`
- `motionscore predict <dataset_root>`
- `motionscore review-init <derivatives_root>`
- `motionscore review-apply <derivatives_root> --scan-id <id> --manual-grade <1..5> --reviewer <name>`
- `motionscore review-clear <derivatives_root> --reviewer <name>`
- `motionscore review-clear <derivatives_root> --all-reviewers`
- `motionscore export <derivatives_root>`
- `motionscore import-final-grades <derivatives_root> --input <tsv_or_csv>`
- `motionscore explain <derivatives_root> --scan-id <id>`
- `motionscore train-prepare <derivatives_root>`
- `motionscore train --manifest <path> --output-model-dir <path>`
- `motionscore model-download --model-id <id>`
- `motionscore model-register --model-id <id> --model-dir <path> --display-name <name>`
- `motionscore model-list`

## Help

Use built-in help for full options:

```bash
motionscore --help
motionscore predict --help
motionscore train-prepare --help
motionscore train --help
motionscore model-download --help
```

## Notes

- `--confidence-threshold` is a review policy setting (`review-init`), not required for pure prediction.
- Model weights are public GitHub release assets. GitHub release asset download counts are the central usage metric.
- `model-download` reads the public model catalog, downloads the selected bundle, verifies `sha256` when present, and registers the extracted checkpoints in `model_registry.json`.
- Retraining is strict fold-aware (`fold_id` required in the manifest).
- Use `--seed` for deterministic sampling/splitting behavior.
- Use `--cv-folds` in `train-prepare` to match ensemble checkpoint count.
