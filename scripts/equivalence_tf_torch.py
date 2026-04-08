#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionscore.inference.model import ModelEnsemble
from motionscore.inference.scoring import predict_scan
from motionscore.io.aim import read_aim


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TensorFlow vs torch prediction parity on AIM scans")
    parser.add_argument("scan_paths", nargs="+", type=Path, help="AIM scan paths")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])
    parser.add_argument("--stackheight", type=int, default=168)
    parser.add_argument("--on-incomplete-stack", type=str, default="keep_last", choices=["keep_last", "drop_last", "error"])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    tf_ensemble = ModelEnsemble(args.model_dir, backend="tensorflow")
    torch_ensemble = ModelEnsemble(args.model_dir, backend="torch")

    pairs = []
    for scan_path in args.scan_paths:
        volume = read_aim(scan_path, scaling=args.scaling)
        tf_pred = predict_scan(
            volume.data,
            ensemble=tf_ensemble,
            stackheight=args.stackheight,
            on_incomplete_stack=args.on_incomplete_stack,
        )
        torch_pred = predict_scan(
            volume.data,
            ensemble=torch_ensemble,
            stackheight=args.stackheight,
            on_incomplete_stack=args.on_incomplete_stack,
        )
        pairs.append(
            {
                "scan_id": str(scan_path),
                "legacy_grade": int(tf_pred.automatic_grade),
                "new_grade": int(torch_pred.automatic_grade),
                "legacy_conf": float(tf_pred.automatic_confidence),
                "new_conf": float(torch_pred.automatic_confidence),
            }
        )
        print(
            f"{scan_path.name}: tf={tf_pred.automatic_grade}/{tf_pred.automatic_confidence} "
            f"torch={torch_pred.automatic_grade}/{torch_pred.automatic_confidence}"
        )

    payload = {"pairs": pairs}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
