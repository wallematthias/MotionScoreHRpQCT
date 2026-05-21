from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pytest

import motionscore.cli as cli
from motionscore.licensing import (
    append_usage_event,
    create_license_record,
    download_and_register_model,
    read_license_record,
    write_license_record,
)
from motionscore.model_registry import resolve_model_dir


def test_create_license_record_and_event_log(tmp_path: Path) -> None:
    record = create_license_record(
        name="Jane Doe",
        institution="UCSF",
        email="JANE@example.edu",
        group="Bone Lab",
        intended_use="Motion scoring",
        tracking_url="https://github.com/example/repo/issues/new",
        created_at="2026-05-20T12:00:00Z",
        license_key="MS-ABCDEF-ABCDEF-ABCDEF-ABCDEF",
    )
    license_path = write_license_record(record, tmp_path / "license.json")
    loaded = read_license_record(license_path)

    assert loaded.email == "jane@example.edu"
    assert loaded.license_key == "MS-ABCDEF-ABCDEF-ABCDEF-ABCDEF"
    assert "MotionScore+registration" in loaded.tracking_submission_url
    assert "UCSF" in loaded.tracking_submission_url

    events_path = append_usage_event(
        event_type="registration",
        payload={"group": loaded.group},
        license_path=license_path,
        events_path=tmp_path / "events.tsv",
    )
    text = events_path.read_text(encoding="utf-8")
    assert "event_type\tlicense_key" in text
    assert "registration" in text
    assert loaded.license_key in text


def test_download_and_register_model_from_catalog(tmp_path: Path) -> None:
    model_bundle = tmp_path / "model.zip"
    with zipfile.ZipFile(model_bundle, "w") as zf:
        zf.writestr("base-v1/DNN_0.pt", b"checkpoint")
        zf.writestr("base-v1/DNN_1.pt", b"checkpoint")

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
                        "url": model_bundle.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    license_path = write_license_record(
        create_license_record(
            name="Jane Doe",
            institution="UCSF",
            email="jane@example.edu",
            created_at="2026-05-20T12:00:00Z",
        ),
        tmp_path / "license.json",
    )

    registry_path, entry = download_and_register_model(
        model_id="base-v1",
        model_root=tmp_path / "models",
        catalog_source=catalog,
        license_path=license_path,
    )

    assert registry_path.exists()
    assert entry["model_id"] == "base-v1"
    resolved_dir, profile = resolve_model_dir(tmp_path / "models", "base-v1")
    assert profile["checkpoint_count"] == 2
    assert (resolved_dir / "DNN_0.pt").exists()
    assert (resolved_dir / "motionscore_download_receipt.json").exists()


def test_download_requires_license(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"default_model_id": "base-v1", "models": [{"model_id": "base-v1", "url": "file:///missing.zip"}]}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        download_and_register_model(
            model_id="base-v1",
            model_root=tmp_path / "models",
            catalog_source=catalog,
            license_path=tmp_path / "missing-license.json",
        )


def test_license_cli_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    license_path = tmp_path / "license.json"
    events_path = tmp_path / "events.tsv"
    args = argparse.Namespace(
        name="Jane Doe",
        institution="UCSF",
        email="jane@example.edu",
        group="Bone Lab",
        intended_use="testing",
        tracking_url="https://github.com/example/repo/issues/new",
        license_path=license_path,
        events_path=events_path,
        open_tracking_url=True,
    )
    monkeypatch.setattr(cli, "open_tracking_submission", lambda _record: True)

    assert cli._cmd_license_register(args) == 0
    assert license_path.exists()
    assert events_path.exists()
    assert "license_key=MS-" in capsys.readouterr().out

    assert cli._cmd_license_show(argparse.Namespace(license_path=license_path, as_json=False)) == 0
    assert "institution=UCSF" in capsys.readouterr().out


def test_license_and_download_parse() -> None:
    parser = cli._build_parser()
    args_license = parser.parse_args(
        [
            "license-register",
            "--name",
            "Jane Doe",
            "--institution",
            "UCSF",
            "--email",
            "jane@example.edu",
        ]
    )
    assert args_license.command == "license-register"
    assert args_license.group == ""

    args_download = parser.parse_args(["model-download", "--model-id", "base-v1"])
    assert args_download.command == "model-download"
    assert args_download.model_id == "base-v1"
