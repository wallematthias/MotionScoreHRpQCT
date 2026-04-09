from pathlib import Path

from motionscore.review.store import (
    apply_manual_review,
    clear_manual_reviews,
    compute_grade_pair_agreement,
    compute_review_agreement,
    export_reviews,
    import_final_grades,
    initialize_or_update_review,
)
from motionscore.utils import read_tsv, write_tsv


def test_review_init_and_manual_override(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    pred = {
        "scan_id": "scan-1",
        "subject_id": "001",
        "site": "tibia",
        "session_id": "T1",
        "automatic_grade": "2",
        "automatic_confidence": "60",
    }

    rows = initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[pred],
        confidence_threshold=75,
    )
    assert len(rows) == 1
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["final_grade"] == ""

    updated = apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        scan_id="scan-1",
        manual_grade=4,
        reviewer="tester",
    )
    assert updated["review_status"] == "manual_override"
    assert updated["final_grade"] == "4"

    on_disk = read_tsv(review_tsv)
    assert on_disk[0]["manual_grade"] == "4"


def test_training_mode_review_flow(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    pred = {
        "scan_id": "scan-1",
        "subject_id": "001",
        "site": "tibia",
        "session_id": "T1",
        "automatic_grade": "3",
        "automatic_confidence": "95",
    }

    rows = initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[pred],
        confidence_threshold=75,
        training_mode=True,
    )
    assert rows[0]["review_status"] == "training_pending"
    assert rows[0]["training_mode"] == "1"
    assert rows[0]["final_grade"] == ""

    updated = apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-1",
        manual_grade=4,
        reviewer="tester",
    )
    assert updated["review_status"] == "training_completed"
    assert updated["manual_grade"] == "4"
    assert updated["prediction_revealed_at"] != ""


def test_compute_review_agreement_metrics() -> None:
    rows = [
        {"automatic_grade": "1", "manual_grade": "1"},
        {"automatic_grade": "2", "manual_grade": "2"},
        {"automatic_grade": "4", "manual_grade": "3"},
        {"automatic_grade": "5", "manual_grade": "5"},
    ]
    stats = compute_review_agreement(rows)
    assert stats["n_scored"] == 4
    assert abs(float(stats["agreement_exact"]) - 0.75) < 1e-9
    assert -1.0 <= float(stats["kappa"]) <= 1.0
    assert -1.0 <= float(stats["kappa_weighted_quadratic"]) <= 1.0


def test_compute_grade_pair_agreement_metrics() -> None:
    pairs = [(1, 1), (2, 2), (3, 4), (5, 5)]
    stats = compute_grade_pair_agreement(pairs)
    assert stats["n_scored"] == 4
    assert abs(float(stats["agreement_exact"]) - 0.75) < 1e-9
    assert -1.0 <= float(stats["kappa"]) <= 1.0
    assert -1.0 <= float(stats["kappa_weighted_quadratic"]) <= 1.0


def test_clear_manual_reviews_by_reviewer(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    pred = {
        "scan_id": "scan-1",
        "subject_id": "001",
        "site": "tibia",
        "session_id": "T1",
        "automatic_grade": "2",
        "automatic_confidence": "60",
    }
    initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[pred],
        confidence_threshold=75,
    )
    apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-1",
        manual_grade=4,
        reviewer="opA",
    )

    cleared = clear_manual_reviews(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        reviewer="opA",
    )
    assert cleared == 1
    row = read_tsv(review_tsv)[0]
    assert row["manual_grade"] == ""
    assert row["reviewer"] == ""
    assert row["review_status"] == "pending"


def test_export_reviews_with_multi_reviewer_consensus(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    pred = {
        "scan_id": "scan-1",
        "subject_id": "001",
        "site": "tibia",
        "session_id": "T1",
        "automatic_grade": "3",
        "automatic_confidence": "80",
    }
    initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[pred],
        confidence_threshold=75,
    )

    apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-1",
        manual_grade=2,
        reviewer="opA",
    )
    apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-1",
        manual_grade=4,
        reviewer="opB",
    )

    output_path = tmp_path / "final_grades.tsv"
    index_rows = [
        {
            "scan_id": "scan-1",
            "raw_image_path": "raw/scan-1.aim",
            "attention_map_path": "",
            "review_tsv": review_tsv.name,
            "review_audit": review_audit.name,
        }
    ]
    export_reviews(index_rows=index_rows, derivatives_root=tmp_path, output_path=output_path)

    exported = read_tsv(output_path)
    assert len(exported) == 1
    row = exported[0]

    assert row["reviewer_count"] == "2"
    assert row["reviewers"] == "opA|opB"
    assert row["consensus_method"] == "mean_manual_grade"
    assert row["consensus_mean_manual_grade"] == "3.000"
    assert row["consensus_grade_rounded"] == "3"

    assert row["reviewer_1_id"] == "opA"
    assert row["reviewer_1_grade"] == "2"
    assert row["reviewer_2_id"] == "opB"
    assert row["reviewer_2_grade"] == "4"


def test_manual_mode_review_flow(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    pred = {
        "scan_id": "scan-1",
        "subject_id": "001",
        "manual_mode": "1",
        "automatic_grade": "",
        "automatic_confidence": "",
    }
    rows = initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[pred],
        confidence_threshold=75,
    )
    assert rows[0]["manual_mode"] == "1"
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["automatic_grade"] == ""

    updated = apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-1",
        manual_grade=3,
        reviewer="opA",
    )
    assert updated["review_status"] == "manual_confirmed"
    assert updated["final_grade"] == "3"


def test_import_final_grades_fills_only_missing_rows(tmp_path: Path) -> None:
    review_tsv = tmp_path / "review.tsv"
    review_json = tmp_path / "review.json"
    review_audit = tmp_path / "review_audit.tsv"

    initialize_or_update_review(
        review_tsv_path=review_tsv,
        review_json_path=review_json,
        review_audit_path=review_audit,
        prediction_rows=[
            {
                "scan_id": "scan-1",
                "subject_id": "001",
                "automatic_grade": "2",
                "automatic_confidence": "60",
            },
            {
                "scan_id": "scan-2",
                "subject_id": "002",
                "automatic_grade": "3",
                "automatic_confidence": "60",
            },
        ],
        confidence_threshold=75,
    )
    apply_manual_review(
        review_tsv_path=review_tsv,
        review_audit_path=review_audit,
        review_json_path=review_json,
        scan_id="scan-2",
        manual_grade=5,
        reviewer="opA",
    )

    import_path = tmp_path / "import.tsv"
    write_tsv(
        import_path,
        [
            {"scan_id": "scan-1", "final_grade": "4", "reviewer": "ext"},
            {"scan_id": "scan-2", "final_grade": "1", "reviewer": "ext"},
            {"scan_id": "missing", "final_grade": "2", "reviewer": "ext"},
        ],
        ["scan_id", "final_grade", "reviewer"],
    )

    stats = import_final_grades(
        index_rows=[
            {
                "scan_id": "scan-1",
                "review_tsv": review_tsv.name,
                "review_audit": review_audit.name,
                "review_json": review_json.name,
            },
            {
                "scan_id": "scan-2",
                "review_tsv": review_tsv.name,
                "review_audit": review_audit.name,
                "review_json": review_json.name,
            },
        ],
        derivatives_root=tmp_path,
        import_path=import_path,
        reviewer="import",
    )

    assert stats == {"imported": 1, "skipped_existing": 1, "missing_scan": 1}
    rows = {row["scan_id"]: row for row in read_tsv(review_tsv)}
    assert rows["scan-1"]["manual_grade"] == "4"
    assert rows["scan-1"]["final_grade"] == "4"
    assert rows["scan-1"]["reviewer"] == "ext"
    assert rows["scan-2"]["manual_grade"] == "5"
