#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from motionscore.inference.model import ModelEnsemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MotionScore Keras .h5 weights to torch .pt weights")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing DNN_*.h5 files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated .pt files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    args = parser.parse_args()

    ensemble = ModelEnsemble(model_dir=args.model_dir, backend="torch")
    converted = ensemble.convert_h5_to_torch(output_dir=args.output_dir, overwrite=args.overwrite)
    print(f"Converted {len(converted)} model(s):")
    for path in converted:
        print(f"  {path}")


if __name__ == "__main__":
    main()
