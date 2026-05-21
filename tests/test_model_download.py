from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import motionscore.cli as cli
from motionscore.licensing import download_and_register_model
from motionscore.model_registry import resolve_model_dir


def _write_catalog(tmp_path: Path, bundle: Path) -> Path:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "default_model_id": "base-v1",
                "models": [
                    {
                        "model_id": "base-v1",
                        "display_name": "Base v1",
                        "version": "v1",
                        "url": bundle.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog


def test_download_and_register_model_from_catalog_without_license(tmp_path: Path) -> None:
    model_bundle = tmp_path / "model.zip"
    with zipfile.ZipFile(model_bundle, "w") as zf:
        zf.writestr("base-v1/DNN_0.pt", b"checkpoint")
        zf.writestr("base-v1/DNN_1.pt", b"checkpoint")
    catalog = _write_catalog(tmp_path, model_bundle)

    registry_path, entry = download_and_register_model(
        model_id="base-v1",
        model_root=tmp_path / "models",
        catalog_source=catalog,
    )

    assert registry_path.exists()
    assert entry["model_id"] == "base-v1"
    resolved_dir, profile = resolve_model_dir(tmp_path / "models", "base-v1")
    assert profile["checkpoint_count"] == 2
    assert (resolved_dir / "DNN_0.pt").exists()
    receipt = json.loads((resolved_dir / "motionscore_download_receipt.json").read_text(encoding="utf-8"))
    assert receipt["model_id"] == "base-v1"
    assert "license_key" not in receipt
    assert "email" not in receipt


def test_download_missing_bundle_reports_download_error(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"default_model_id": "base-v1", "models": [{"model_id": "base-v1", "url": "file:///missing.zip"}]}),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        download_and_register_model(
            model_id="base-v1",
            model_root=tmp_path / "models",
            catalog_source=catalog,
        )


def test_model_download_parse_has_no_license_options() -> None:
    parser = cli._build_parser()
    args_download = parser.parse_args(["model-download", "--model-id", "base-v1"])
    assert args_download.command == "model-download"
    assert args_download.model_id == "base-v1"
    assert not hasattr(args_download, "license_path")

    with pytest.raises(SystemExit):
        parser.parse_args(["license-register", "--name", "Jane Doe"])
