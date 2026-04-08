from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motionscore.io import aim as aim_io


def test_processing_log_as_text_variants() -> None:
    assert aim_io._processing_log_as_text({"processing_log_raw": "abc"}) == "abc"
    assert aim_io._processing_log_as_text({"processing_log": {"A": 1, "B": 2}}) in {
        "A: 1\nB: 2",
        "B: 2\nA: 1",
    }


def test_get_aim_calibration_constants_parses() -> None:
    log = "\n".join(
        [
            "Mu_Scaling 8192",
            "HU: mu water 0.24000",
            "Density: slope 1500.0",
            "Density: intercept -10.5",
        ]
    )
    out = aim_io._get_aim_calibration_constants_from_processing_log(log)
    assert out == (8192, 0.24, 0.0, 1500.0, -10.5)


def test_get_aim_calibration_constants_raises_on_missing() -> None:
    with pytest.raises(ValueError):
        aim_io._get_aim_calibration_constants_from_processing_log("Mu_Scaling 8192")


def test_apply_scaling_modes() -> None:
    log = "\n".join(
        [
            "Mu_Scaling 1000",
            "HU: mu water 0.25",
            "Density: slope 2.0",
            "Density: intercept 1.0",
        ]
    )
    arr = np.asarray([[[1000.0]]], dtype=np.float32)

    native, unit_native = aim_io._apply_scaling(arr, log, "native")
    assert unit_native == "native"
    assert np.allclose(native, arr)

    mu, unit_mu = aim_io._apply_scaling(arr, log, "mu")
    assert unit_mu == "mu"
    assert np.allclose(mu, [[[1.0]]])

    hu, unit_hu = aim_io._apply_scaling(arr, log, "hu")
    assert unit_hu == "hu"
    assert hu.shape == arr.shape

    bmd, unit_bmd = aim_io._apply_scaling(arr, log, "bmd")
    assert unit_bmd == "bmd"
    assert np.allclose(bmd, [[[3.0]]])

    with pytest.raises(ValueError):
        aim_io._apply_scaling(arr, log, "invalid")


def test_as_zyx_and_origin_resolution() -> None:
    arr_zyx = np.zeros((3, 2, 1), dtype=np.int16)
    assert aim_io._as_zyx(arr_zyx, (1, 2, 3)).shape == (3, 2, 1)

    arr_xyz = np.zeros((1, 2, 3), dtype=np.int16)
    converted = aim_io._as_zyx(arr_xyz, (1, 2, 3))
    assert converted.shape == (3, 2, 1)

    spacing = (0.1, 0.2, 0.3)
    assert aim_io._resolve_origin({"origin": (1, 2, 3)}, spacing) == (1.0, 2.0, 3.0)
    pos_origin = aim_io._resolve_origin({"position": (10, 20, 30), "offset": (1, 2, 3)}, spacing)
    assert np.allclose(pos_origin, ((11.5 * 0.1), (22.5 * 0.2), (33.5 * 0.3)))
    assert aim_io._resolve_origin({}, spacing) == (0.0, 0.0, 0.0)


def test_load_py_aimio_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str):
        raise ImportError("no py_aimio")

    monkeypatch.setattr(aim_io.importlib, "import_module", _raise)
    with pytest.raises(RuntimeError, match="py_aimio is required"):
        aim_io._load_py_aimio()


def test_read_aim_with_mock_py_aimio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # array is xyz here; code should transpose to xyz output consistently
    arr_xyz = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    meta = {
        "dimensions": (2, 3, 4),
        "element_size": (0.061, 0.061, 0.082),
        "processing_log_raw": "\n".join(
            [
                "Mu_Scaling 1000",
                "HU: mu water 0.25",
                "Density: slope 2.0",
                "Density: intercept 1.0",
            ]
        ),
        "origin": (1.0, 2.0, 3.0),
    }

    fake_mod = SimpleNamespace(read_aim=lambda _p, density=False, hu=False: (arr_xyz, meta))
    monkeypatch.setattr(aim_io.importlib, "import_module", lambda _name: fake_mod)

    path = tmp_path / "x.AIM"
    path.write_bytes(b"dummy")
    vol = aim_io.read_aim(path, scaling="density")
    assert vol.data.shape == (2, 3, 4)
    assert vol.unit == "bmd"
    assert vol.spacing == (0.061, 0.061, 0.082)
    assert vol.origin == (1.0, 2.0, 3.0)
    assert "Mu_Scaling" in vol.processing_log


def test_write_volume_mha_uses_simpleitk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    written = {}

    class _FakeImage:
        def __init__(self):
            self.spacing = None
            self.origin = None
            self.direction = None

        def SetSpacing(self, v):
            self.spacing = tuple(v)

        def SetOrigin(self, v):
            self.origin = tuple(v)

        def SetDirection(self, v):
            self.direction = tuple(v)

    def _get_image_from_array(arr):
        written["shape_zyx"] = tuple(arr.shape)
        return _FakeImage()

    def _write_image(image, out_path):
        written["path"] = out_path
        written["spacing"] = image.spacing
        written["origin"] = image.origin
        written["direction"] = image.direction

    fake_sitk = SimpleNamespace(
        GetImageFromArray=_get_image_from_array,
        WriteImage=_write_image,
    )
    monkeypatch.setitem(__import__("sys").modules, "SimpleITK", fake_sitk)

    out = tmp_path / "a" / "att.mha"
    vol = np.zeros((4, 3, 2), dtype=np.float32)  # xyz
    returned = aim_io.write_volume_mha(
        out,
        vol,
        spacing=(0.1, 0.2, 0.3),
        origin=(1.0, 2.0, 3.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert returned == out
    assert written["shape_zyx"] == (2, 3, 4)
    assert written["spacing"] == (0.1, 0.2, 0.3)
    assert written["origin"] == (1.0, 2.0, 3.0)
