#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from secrets import token_bytes


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")
    return files


def _build_tar_bytes(base_dir: Path, files: list[Path]) -> bytes:
    bio = BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for path in files:
            arcname = path.relative_to(base_dir).as_posix()
            tf.add(path, arcname=arcname)
    return bio.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create encrypted model bundle + manifest for MotionScore worker delivery.")
    parser.add_argument("--input-dir", type=Path, default=Path("models"), help="Directory containing model files")
    parser.add_argument("--pattern", type=str, default="DNN_*.pt", help="Glob pattern for files")
    parser.add_argument("--version", type=str, required=True, help="Model version label, e.g. v1")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for .enc and manifest")
    parser.add_argument(
        "--key-base64",
        type=str,
        default="",
        help="Optional existing AES-256 key in base64. If omitted, a new random key is generated.",
    )
    args = parser.parse_args()

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'cryptography'. Install with: pip install cryptography"
        ) from exc

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(input_dir, args.pattern)
    bundle_plain = _build_tar_bytes(input_dir, files)

    if args.key_base64.strip():
        key = base64.b64decode(args.key_base64.strip())
        if len(key) != 32:
            raise ValueError("key-base64 must decode to 32 bytes (AES-256)")
        key_generated = False
    else:
        key = token_bytes(32)
        key_generated = True

    nonce = token_bytes(12)
    aad = args.version.encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, bundle_plain, aad)

    enc_name = f"{args.version}.enc"
    manifest_name = f"{args.version}.manifest.json"
    enc_path = output_dir / enc_name
    manifest_path = output_dir / manifest_name

    enc_path.write_bytes(ciphertext)

    manifest = {
        "version": args.version,
        "algorithm": "AES-256-GCM",
        "nonce_base64": base64.b64encode(nonce).decode("ascii"),
        "aad_base64": base64.b64encode(aad).decode("ascii"),
        "encrypted_filename": enc_name,
        "encrypted_size_bytes": len(ciphertext),
        "encrypted_sha256": _sha256_hex(ciphertext),
        "plaintext_sha256": _sha256_hex(bundle_plain),
        "file_pattern": args.pattern,
        "file_count": len(files),
        "files": [str(p.relative_to(input_dir).as_posix()) for p in files],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote encrypted bundle: {enc_path}")
    print(f"Wrote manifest: {manifest_path}")
    print("")
    print("Set this as Worker secret MODEL_MASTER_KEY (base64):")
    print(base64.b64encode(key).decode("ascii"))
    if key_generated:
        print("")
        print("A new key was generated in this run. Save it in a secure secret manager.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
