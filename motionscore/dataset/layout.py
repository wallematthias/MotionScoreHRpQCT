from __future__ import annotations

from pathlib import Path

from motionscore.dataset.models import RawSession


PIPELINE_NAME = "MotionScore"


def get_derivatives_root(input_root: str | Path, output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
        if root.name == PIPELINE_NAME:
            return root
        return root / PIPELINE_NAME

    input_root = Path(input_root)
    derivatives = input_root / "derivatives" / PIPELINE_NAME
    return derivatives


def get_subject_dir(derivatives_root: str | Path, subject_id: str) -> Path:
    return Path(derivatives_root) / f"sub-{subject_id}"


def get_site_dir(derivatives_root: str | Path, subject_id: str, site: str) -> Path:
    return get_subject_dir(derivatives_root, subject_id) / f"site-{site}"


def get_session_dir(derivatives_root: str | Path, session: RawSession) -> Path:
    if session.output_rel_dir:
        return Path(derivatives_root) / session.output_rel_dir
    return get_site_dir(derivatives_root, session.subject_id, session.site) / f"ses-{session.session_id}"


def get_predictions_dir(derivatives_root: str | Path, session: RawSession) -> Path:
    return get_session_dir(derivatives_root, session) / "predictions"


def get_review_dir(derivatives_root: str | Path, session: RawSession) -> Path:
    return get_session_dir(derivatives_root, session) / "review"


def get_explain_dir(derivatives_root: str | Path, session: RawSession) -> Path:
    return get_session_dir(derivatives_root, session) / "explain"


def get_preview_dir(derivatives_root: str | Path, session: RawSession) -> Path:
    return get_session_dir(derivatives_root, session) / "preview"


def get_index_path(derivatives_root: str | Path) -> Path:
    return Path(derivatives_root) / "index.tsv"


def get_dataset_description_path(derivatives_root: str | Path) -> Path:
    return Path(derivatives_root) / "dataset_description.json"


def get_export_path(derivatives_root: str | Path) -> Path:
    return Path(derivatives_root) / "motion_grades.tsv"
