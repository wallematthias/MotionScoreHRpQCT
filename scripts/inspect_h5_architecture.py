#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def summarize_model(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise ValueError(f"model_config not found in {h5_path}")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        model_config = json.loads(raw)

    layers = []
    for layer in model_config.get("config", {}).get("layers", []):
        cfg = layer.get("config", {})
        layers.append(
            {
                "class_name": layer.get("class_name"),
                "name": cfg.get("name"),
                "activation": cfg.get("activation"),
                "kernel_size": cfg.get("kernel_size"),
                "strides": cfg.get("strides"),
                "padding": cfg.get("padding"),
                "filters": cfg.get("filters"),
                "units": cfg.get("units"),
                "pool_size": cfg.get("pool_size"),
            }
        )
    return {"path": str(h5_path.resolve()), "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Keras .h5 architecture config")
    parser.add_argument("model_paths", nargs="+", type=Path, help="One or more .h5 model files")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    summaries = [summarize_model(p) for p in args.model_paths]
    payload = {"models": summaries}

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
