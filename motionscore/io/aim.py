from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import re

import numpy as np


@dataclass(slots=True)
class AimVolume:
    data: np.ndarray  # x, y, z
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    processing_log: str
    unit: str


def _load_py_aimio():
    try:
        return importlib.import_module("py_aimio")
    except ImportError as exc:
        raise RuntimeError(
            "py_aimio is required for AIM file reading; install package 'aimio-py'."
        ) from exc


def _processing_log_as_text(meta: dict) -> str:
    log = meta.get("processing_log_raw", meta.get("processing_log", ""))
    if isinstance(log, dict):
        lines = [f"{k}: {v}" for k, v in log.items()]
        return "\n".join(lines)
    return str(log)


def _get_aim_calibration_constants_from_processing_log(
    processing_log: str,
) -> tuple[int, float, float, float, float]:
    mu_scaling_match = re.search(r"Mu_Scaling\s+(\d+)", processing_log)
    hu_mu_water_match = re.search(r"HU: mu water\s+(\d+\.\d+)", processing_log)
    density_slope_match = re.search(
        r"Density: slope\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )
    density_intercept_match = re.search(
        r"Density: intercept\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )

    if not all(
        [
            mu_scaling_match,
            hu_mu_water_match,
            density_slope_match,
            density_intercept_match,
        ]
    ):
        raise ValueError("Could not parse AIM calibration constants from processing log")

    mu_scaling = int(mu_scaling_match.group(1))
    hu_mu_water = float(hu_mu_water_match.group(1))
    hu_mu_air = 0.0
    density_slope = float(density_slope_match.group(1))
    density_intercept = float(density_intercept_match.group(1))
    return mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept


def _apply_scaling(np_image: np.ndarray, processing_log: str, scaling: str) -> tuple[np.ndarray, str]:
    scaling = scaling.lower()
    if scaling in {"native", "none"}:
        return np_image, "native"

    mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
        _get_aim_calibration_constants_from_processing_log(processing_log)
    )

    if scaling == "mu":
        return np_image.astype(np.float32) / float(mu_scaling), "mu"

    if scaling == "hu":
        m = 1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))
        b = -1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
        return np_image.astype(np.float32) * m + b, "hu"

    if scaling in {"bmd", "density"}:
        out = np_image.astype(np.float32) / float(mu_scaling) * float(density_slope) + float(
            density_intercept
        )
        return out, "bmd"

    raise ValueError(
        f"Unsupported scaling '{scaling}'. Use one of: native, none, mu, hu, bmd, density."
    )


def _as_zyx(array: np.ndarray, dimensions_xyz: tuple[int, int, int] | None) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D AIM array, got shape {array.shape}.")

    if dimensions_xyz is None:
        return array

    expected_zyx = (dimensions_xyz[2], dimensions_xyz[1], dimensions_xyz[0])
    expected_xyz = dimensions_xyz

    if tuple(array.shape) == expected_zyx:
        return array
    if tuple(array.shape) == expected_xyz:
        return np.transpose(array, (2, 1, 0))
    return array


def _resolve_origin(meta: dict, spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    origin_raw = meta.get("origin")
    if isinstance(origin_raw, (list, tuple)) and len(origin_raw) == 3:
        return tuple(float(v) for v in origin_raw)

    position_raw = meta.get("position")
    if isinstance(position_raw, (list, tuple)) and len(position_raw) == 3:
        offset_raw = meta.get("offset", (0, 0, 0))
        if not (isinstance(offset_raw, (list, tuple)) and len(offset_raw) == 3):
            offset_raw = (0, 0, 0)
        return tuple(
            (float(position_raw[i]) + float(offset_raw[i]) + 0.5) * float(spacing[i])
            for i in range(3)
        )

    return (0.0, 0.0, 0.0)


def read_aim(path: Path, scaling: str = "native") -> AimVolume:
    py_aimio = _load_py_aimio()
    arr, meta = py_aimio.read_aim(str(path), density=False, hu=False)
    meta = dict(meta)

    dims_xyz_raw = meta.get("dimensions")
    if isinstance(dims_xyz_raw, (list, tuple)) and len(dims_xyz_raw) == 3:
        dims_xyz = tuple(int(v) for v in dims_xyz_raw)
    else:
        dims_xyz = None

    arr_zyx = _as_zyx(np.asarray(arr), dims_xyz)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

    processing_log = _processing_log_as_text(meta)
    arr_scaled, unit = _apply_scaling(arr_xyz, processing_log, scaling)

    spacing_raw = meta.get("element_size", meta.get("spacing", (1.0, 1.0, 1.0)))
    if isinstance(spacing_raw, (list, tuple)) and len(spacing_raw) == 3:
        spacing = tuple(float(v) for v in spacing_raw)
    else:
        spacing = (1.0, 1.0, 1.0)

    origin = _resolve_origin(meta, spacing)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    return AimVolume(
        data=arr_scaled,
        spacing=spacing,
        origin=origin,
        direction=direction,
        processing_log=processing_log,
        unit=unit,
    )


def write_volume_mha(
    path: Path,
    volume_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: tuple[float, ...] | None = None,
) -> Path:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to write .mha attention maps") from exc

    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume_xyz, got shape {volume_xyz.shape}")

    arr_zyx = np.transpose(volume_xyz, (2, 1, 0))
    image = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))
    if direction is not None:
        image.SetDirection(direction)

    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path
