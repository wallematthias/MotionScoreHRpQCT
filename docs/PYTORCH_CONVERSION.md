# PyTorch Conversion Workflow

This document describes the implemented TensorFlow-to-PyTorch migration path for MotionScore models.

## 1) Inspect source `.h5` architecture

```bash
python scripts/inspect_h5_architecture.py models/DNN_0.h5 --output-json /tmp/motionscore_architecture.json
```

This captures layer-level architecture metadata from Keras configs.

## 2) Convert `.h5` ensemble to `.pt`

Use either CLI command:

```bash
motionscore convert-torch --model-dir ./models --output-dir ./models
```

or script:

```bash
python scripts/convert_h5_to_torch.py --model-dir ./models --output-dir ./models
```

Expected outputs:

```text
models/DNN_0.pt
models/DNN_1.pt
...
models/DNN_9.pt
```

## 3) Run inference with torch backend

```bash
motionscore predict /path/to/dataset --backend torch
motionscore explain /path/to/dataset/derivatives/MotionScore --scan-id <scan_id> --backend torch
```

## 4) Run TF-vs-Torch equivalence report

```bash
python scripts/equivalence_tf_torch.py \
  /path/to/scan1.AIM /path/to/scan2.AIM \
  --model-dir ./models \
  --output-json /tmp/motionscore_tf_vs_torch.json
```

You can point `MOTIONSCORE_EQUIVALENCE_JSON` at the generated JSON to activate the strict gate in `tests/test_equivalence_gate.py`.

## Notes

- Torch backend supports loading both `.pt` and `.h5` model paths directly.
- `.h5` loading in torch backend uses direct weight mapping (no TensorFlow runtime required).
- TensorFlow backend remains available for parity validation during migration.
