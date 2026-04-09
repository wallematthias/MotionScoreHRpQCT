from pathlib import Path

import pytest

from motionscore.model_registry import list_model_profiles, register_model_profile, resolve_model_dir


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")


def test_register_and_resolve_model_profile(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_dir = model_root / "base-v1"
    _touch(model_dir / "DNN_0.pt")
    _touch(model_dir / "DNN_1.pt")

    registry_path, entry = register_model_profile(
        model_root=model_root,
        model_id="base-v1",
        model_dir=model_dir,
        display_name="Base v1",
        domain="hrpqct",
        version="v1",
        make_default=True,
    )
    assert registry_path.exists()
    assert entry["model_id"] == "base-v1"

    resolved_dir, profile = resolve_model_dir(model_root=model_root, model_id="base-v1")
    assert resolved_dir == model_dir.resolve()
    assert profile["model_id"] == "base-v1"

    profiles = list_model_profiles(model_root=model_root)
    assert len(profiles) == 1
    assert profiles[0]["display_name"] == "Base v1"


def test_resolve_model_dir_errors_on_unknown_id(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_dir = model_root / "base-v1"
    _touch(model_dir / "DNN_0.pt")

    register_model_profile(
        model_root=model_root,
        model_id="base-v1",
        model_dir=model_dir,
        display_name="Base v1",
        domain="hrpqct",
        version="v1",
        make_default=True,
    )

    with pytest.raises(KeyError):
        resolve_model_dir(model_root=model_root, model_id="knee-v1")


def test_resolve_model_dir_implicit_root_profile(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    _touch(model_root / "DNN_0.pt")
    _touch(model_root / "DNN_1.pt")

    resolved_dir, profile = resolve_model_dir(model_root=model_root, model_id="base-v1")
    assert resolved_dir == model_root.resolve()
    assert profile["model_id"] == "base-v1"

    profiles = list_model_profiles(model_root=model_root)
    assert len(profiles) == 1
    assert profiles[0]["relative_dir"] == "."
