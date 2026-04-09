from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import motionscore.training.prepare as prepare_module
from motionscore.training.prepare import build_training_manifest
from motionscore.utils import read_tsv, write_tsv


def _write_scan_tables(derivatives: Path, *, rel_base: str, scan_id: str, subject_id: str, manual_grade: str, slice_grades: str, slice_conf: str) -> None:
    base = derivatives / rel_base
    pred = base / "predictions" / "predictions.tsv"
    review = base / "review" / "review.tsv"

    write_tsv(
        pred,
        [
            {
                "scan_id": scan_id,
                "subject_id": subject_id,
                "raw_image_path": f"/tmp/{scan_id}.AIM",
                "slice_grades": slice_grades,
                "slice_confidences": slice_conf,
                "automatic_grade": "3",
                "automatic_confidence": "82",
            }
        ],
        [
            "scan_id",
            "subject_id",
            "raw_image_path",
            "slice_grades",
            "slice_confidences",
            "automatic_grade",
            "automatic_confidence",
        ],
    )

    write_tsv(
        review,
        [
            {
                "scan_id": scan_id,
                "subject_id": subject_id,
                "manual_grade": manual_grade,
            }
        ],
        ["scan_id", "subject_id", "manual_grade"],
    )


def test_build_training_manifest_manual_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    derivatives = tmp_path / "MotionScore"
    derivatives.mkdir(parents=True)
    monkeypatch.setattr(
        prepare_module,
        "read_aim",
        lambda _path, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)),
    )
    monkeypatch.setattr(
        prepare_module,
        "preprocess_slice",
        lambda arr: (np.asarray(arr, dtype=np.uint8), {}),
    )

    _write_scan_tables(
        derivatives,
        rel_base="sub-S1/site-tibia/ses-T1",
        scan_id="scan_manual",
        subject_id="S1",
        manual_grade="4",
        slice_grades="[1,2,3]",
        slice_conf="[0.2,0.3,0.4]",
    )
    _write_scan_tables(
        derivatives,
        rel_base="sub-S2/site-tibia/ses-T1",
        scan_id="scan_auto",
        subject_id="S2",
        manual_grade="",
        slice_grades="[2,0,5]",
        slice_conf="[0.8,-1,0.6]",
    )

    write_tsv(
        derivatives / "index.tsv",
        [
            {
                "scan_id": "scan_manual",
                "subject_id": "S1",
                "raw_image_path": "/tmp/scan_manual.AIM",
                "predictions_tsv": "sub-S1/site-tibia/ses-T1/predictions/predictions.tsv",
                "review_tsv": "sub-S1/site-tibia/ses-T1/review/review.tsv",
            },
            {
                "scan_id": "scan_auto",
                "subject_id": "S2",
                "raw_image_path": "/tmp/scan_auto.AIM",
                "predictions_tsv": "sub-S2/site-tibia/ses-T1/predictions/predictions.tsv",
                "review_tsv": "sub-S2/site-tibia/ses-T1/review/review.tsv",
            },
        ],
        ["scan_id", "subject_id", "raw_image_path", "predictions_tsv", "review_tsv"],
    )

    out_path = derivatives / "training" / "train_manifest.tsv"
    stats = build_training_manifest(
        derivatives_root=derivatives,
        output_path=out_path,
        min_auto_confidence=0.7,
        slice_step=1,
        include_auto_without_manual=True,
        seed=13,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    rows = read_tsv(out_path)
    manual_rows = [r for r in rows if r.get("scan_id") == "scan_manual"]
    auto_rows = [r for r in rows if r.get("scan_id") == "scan_auto"]

    assert len(manual_rows) == 3
    assert all(r.get("label") == "4" for r in manual_rows)
    assert all(r.get("label_source") == "manual_scan_propagated" for r in manual_rows)

    assert len(auto_rows) == 1
    assert auto_rows[0]["slice_index"] == "0"
    assert auto_rows[0]["label"] == "2"
    assert auto_rows[0]["label_source"] == "auto_slice"
    assert all(str(r.get("cache_npy_path", "")).strip() for r in rows)
    assert all(str(r.get("cache_index", "")).strip() for r in rows)

    assert stats["rows_manual"] == 3
    assert stats["rows_auto"] == 1
    assert stats["rows_written"] == 4
    assert stats["cache_scans"] == 2
    assert stats["cache_slices"] == 4


def test_build_training_manifest_slice_count_is_seeded_and_randomized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    derivatives = tmp_path / "MotionScore"
    derivatives.mkdir(parents=True)
    monkeypatch.setattr(
        prepare_module,
        "read_aim",
        lambda _path, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 16), dtype=np.float32)),
    )
    monkeypatch.setattr(
        prepare_module,
        "preprocess_slice",
        lambda arr: (np.asarray(arr, dtype=np.uint8), {}),
    )

    slice_grades = "[" + ",".join(["3"] * 16) + "]"
    slice_conf = "[" + ",".join(["0.9"] * 16) + "]"
    _write_scan_tables(
        derivatives,
        rel_base="sub-S1/site-tibia/ses-T1",
        scan_id="scan_seeded",
        subject_id="S1",
        manual_grade="4",
        slice_grades=slice_grades,
        slice_conf=slice_conf,
    )
    write_tsv(
        derivatives / "index.tsv",
        [
            {
                "scan_id": "scan_seeded",
                "subject_id": "S1",
                "raw_image_path": "/tmp/scan_seeded.AIM",
                "predictions_tsv": "sub-S1/site-tibia/ses-T1/predictions/predictions.tsv",
                "review_tsv": "sub-S1/site-tibia/ses-T1/review/review.tsv",
            },
        ],
        ["scan_id", "subject_id", "raw_image_path", "predictions_tsv", "review_tsv"],
    )

    out_a = derivatives / "training" / "manifest_a.tsv"
    out_b = derivatives / "training" / "manifest_b.tsv"
    out_c = derivatives / "training" / "manifest_c.tsv"

    build_training_manifest(
        derivatives_root=derivatives,
        output_path=out_a,
        min_auto_confidence=0.7,
        slice_step=1,
        slice_count=8,
        include_auto_without_manual=False,
        seed=13,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    build_training_manifest(
        derivatives_root=derivatives,
        output_path=out_b,
        min_auto_confidence=0.7,
        slice_step=1,
        slice_count=8,
        include_auto_without_manual=False,
        seed=13,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    build_training_manifest(
        derivatives_root=derivatives,
        output_path=out_c,
        min_auto_confidence=0.7,
        slice_step=1,
        slice_count=8,
        include_auto_without_manual=False,
        seed=99,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    z_a = [int(r["slice_index"]) for r in read_tsv(out_a)]
    z_b = [int(r["slice_index"]) for r in read_tsv(out_b)]
    z_c = [int(r["slice_index"]) for r in read_tsv(out_c)]

    assert len(z_a) == 8
    assert z_a == z_b
    assert z_c != z_a
