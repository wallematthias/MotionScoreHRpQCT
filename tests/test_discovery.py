from pathlib import Path

from motionscore.config import DiscoveryConfig
from motionscore.dataset.discovery import discover_raw_sessions, _infer_role_from_processing_log


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_discover_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    root.mkdir(parents=True, exist_ok=True)
    _touch(root / "SUB001_DT_T1.AIM")
    _touch(root / "SUB001_DT_T1_TRAB_MASK.AIM")

    sessions = discover_raw_sessions(root, DiscoveryConfig())
    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUB001"
    assert sessions[0].site == "tibia"
    assert sessions[0].session_id == "T1"
    assert sessions[0].output_rel_dir == Path("SUB001_DT_T1")


def test_discover_nested_bids_layout(tmp_path: Path) -> None:
    image = tmp_path / "sub-001" / "ses-T2" / "anat" / "SUB001_DT_T2.AIM"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("", encoding="utf-8")

    sessions = discover_raw_sessions(tmp_path, DiscoveryConfig())
    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUB001"
    assert sessions[0].site == "tibia"
    assert sessions[0].session_id == "T2"
    assert sessions[0].output_rel_dir == Path("sub-001") / "ses-T2" / "anat"


def test_discover_ignores_derivatives_copy(tmp_path: Path) -> None:
    raw = tmp_path / "SUB001_DT_T1.AIM"
    raw.write_text("", encoding="utf-8")

    copied = (
        tmp_path
        / "derivatives"
        / "MotionScore"
        / "sub-SUB001"
        / "site-tibia"
        / "ses-T1"
        / "SUB001_DT_T1.AIM"
    )
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_text("", encoding="utf-8")

    sessions = discover_raw_sessions(tmp_path, DiscoveryConfig())
    assert len(sessions) == 1
    assert sessions[0].raw_image_path == raw


def test_infer_role_from_processing_log_raw_isq_markers() -> None:
    log = """
    Created by                    ISQ_TO_AIM (IPL)
    Original file                 xxxxx.isq;
    Orig-ISQ-Dim-p                2304 2304 168
    """
    assert _infer_role_from_processing_log(log) == "image"


def test_infer_role_from_processing_log_mask_markers() -> None:
    log = """
    Created by                    D3P_GobjCreateAimPeel (IPL)
    Time                           2-NOV-2022 20:54:42.04
    """
    assert _infer_role_from_processing_log(log) == "derived"
