# MotionScore Slicer Integration Contract

This repository provides core pipeline logic only.

A separate Slicer extension repository should use the following contract.

## Primary UX

- `Run` button
  - Calls `motionscore predict <dataset_root> [--output-root ...] [--confidence-threshold ...] [--training-mode]`
  - Produces derivatives under `derivatives/MotionScore`
  - Optional quick-QC snapshots are emitted during predict:
    - `preview/<scan_id>_preview.png`
    - `preview/<scan_id>_slice_profile.png`

- `Review` button
  - Calls `motionscore review-init <derivatives_root> [--confidence-threshold ...] [--training-mode]`
  - Reads pending scans from per-session `review.tsv`
  - Default mode: show auto grade + confidence
  - Training mode: hide AI prediction until operator submits a manual grade, then reveal AI grade
  - Reviewer confirms or overwrites by calling:
    - `motionscore review-apply <derivatives_root> --scan-id ... --manual-grade ... --reviewer ...`
  - Re-review workflows can clear prior manual grades:
    - `motionscore review-clear <derivatives_root> --reviewer <operator_id>`
    - `motionscore review-clear <derivatives_root> --all-reviewers`
  - Display live agreement metrics (exact agreement + Cohen kappa + weighted kappa) from current review rows
  - Display pairwise agreement matrix between AI and all known reviewers (from audit history)

- Attention overlay (on demand)
  - When reviewer opens a scan, request map generation only if missing (or if overwrite requested):
    - `motionscore explain <derivatives_root> --scan-id ...`
  - Load `<scan_id>_gradcam.mha` as overlay in Slicer

- Export table
  - `motionscore export <derivatives_root> [--output ...]`
  - Export includes per-scan multi-reviewer summary fields:
    - reviewer slots (`reviewer_N_id`, `reviewer_N_grade`)
    - `reviewer_count`, `reviewers`
    - consensus fields (`consensus_method`, `consensus_mean_manual_grade`, `consensus_grade_rounded`)

## Data Ownership

- Slicer must not create ad hoc sidecar state files.
- Persisted truth lives in derivatives managed by core CLI:
  - `predictions/predictions.tsv`
  - `preview/*_preview.png`
  - `preview/*_slice_profile.png`
  - `review/review.tsv`
  - `review/review.json`
  - `review/review_audit.tsv`
  - `explain/*_gradcam.mha`

## Scan Identity

Use `scan_id` from `index.tsv` as the stable key for all review/apply/explain actions.
