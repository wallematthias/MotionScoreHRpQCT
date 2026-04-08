from pathlib import Path

from motionscore.dataset.layout import get_derivatives_root, get_preview_dir
from motionscore.dataset.models import RawSession


def test_derivatives_default_root() -> None:
    root = Path("/tmp/my_dataset")
    assert get_derivatives_root(root) == root / "derivatives" / "MotionScore"


def test_derivatives_custom_output_root() -> None:
    root = Path("/tmp/my_dataset")
    out = Path("/tmp/results")
    assert get_derivatives_root(root, out) == out / "MotionScore"


def test_derivatives_custom_pipeline_root_passthrough() -> None:
    root = Path("/tmp/my_dataset")
    out = Path("/tmp/results/MotionScore")
    assert get_derivatives_root(root, out) == out


def test_preview_dir_layout() -> None:
    session = RawSession(
        subject_id="001",
        site="tibia",
        session_id="T1",
        raw_image_path=Path("/tmp/raw.aim"),
    )
    derivatives = Path("/tmp/my_dataset/derivatives/MotionScore")
    expected = derivatives / "sub-001" / "site-tibia" / "ses-T1" / "preview"
    assert get_preview_dir(derivatives, session) == expected


def test_preview_dir_layout_with_output_rel_dir() -> None:
    session = RawSession(
        subject_id="001",
        site="tibia",
        session_id="T1",
        raw_image_path=Path("/tmp/raw.aim"),
        output_rel_dir=Path("flat_scan_001"),
    )
    derivatives = Path("/tmp/my_dataset/derivatives/MotionScore")
    expected = derivatives / "flat_scan_001" / "preview"
    assert get_preview_dir(derivatives, session) == expected
