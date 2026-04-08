from __future__ import annotations

from collections import Counter
from pathlib import Path

from motionscore.utils import read_tsv, utc_now_iso, write_json, write_tsv

REVIEW_FIELDS = [
    "scan_id",
    "subject_id",
    "site",
    "session_id",
    "automatic_grade",
    "automatic_confidence",
    "confidence_threshold",
    "manual_grade",
    "final_grade",
    "review_status",
    "training_mode",
    "prediction_revealed_at",
    "reviewer",
    "reviewed_at",
    "last_updated",
]

AUDIT_FIELDS = [
    "timestamp",
    "scan_id",
    "event",
    "automatic_grade",
    "manual_grade",
    "final_grade",
    "reviewer",
    "notes",
]


def _to_int(text: str | int | None, default: int = 0) -> int:
    if text is None:
        return default
    if isinstance(text, int):
        return text
    text = str(text).strip()
    if not text:
        return default
    return int(float(text))


def _to_bool(text: str | int | bool | None, default: bool = False) -> bool:
    if text is None:
        return default
    if isinstance(text, bool):
        return text
    if isinstance(text, int):
        return text != 0
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _read_review_json_settings(review_json_path: Path | None) -> tuple[int, bool]:
    threshold = 75
    training_mode = False
    if review_json_path is None:
        return threshold, training_mode
    try:
        from json import loads

        raw = review_json_path.read_text(encoding="utf-8").strip()
        if raw:
            payload = loads(raw)
            threshold = int(payload.get("confidence_threshold", threshold))
            training_mode = bool(payload.get("training_mode", training_mode))
    except Exception:
        pass
    return threshold, training_mode


def _write_review_summary_json(
    review_json_path: Path | None,
    rows: list[dict[str, str]],
    confidence_threshold: int,
    training_mode: bool,
    updated_at: str,
) -> None:
    if review_json_path is None:
        return
    agreement = compute_review_agreement(rows)
    write_json(
        review_json_path,
        {
            "schema_version": "1.1",
            "confidence_threshold": int(confidence_threshold),
            "training_mode": bool(training_mode),
            "updated_at": updated_at,
            "n_scans": len(rows),
            "agreement": agreement,
        },
    )


def compute_grade_pair_agreement(pairs: list[tuple[int, int]]) -> dict[str, float | int]:
    n_scored = len(pairs)
    if n_scored == 0:
        return {
            "n_scored": 0,
            "agreement_exact": 0.0,
            "kappa": 0.0,
            "kappa_weighted_quadratic": 0.0,
        }

    exact_matches = sum(1 for auto_grade, manual_grade in pairs if auto_grade == manual_grade)
    agreement_exact = float(exact_matches) / float(n_scored)

    auto_counts = Counter(auto_grade for auto_grade, _ in pairs)
    manual_counts = Counter(manual_grade for _, manual_grade in pairs)
    expected = 0.0
    for grade in range(1, 6):
        expected += (auto_counts[grade] / n_scored) * (manual_counts[grade] / n_scored)
    if abs(1.0 - expected) < 1e-12:
        kappa = 1.0 if agreement_exact >= 1.0 else 0.0
    else:
        kappa = (agreement_exact - expected) / (1.0 - expected)

    # Quadratic weighted kappa for ordinal grading.
    denom = 0.0
    numer = 0.0
    n_minus_1_sq = float((5 - 1) ** 2)
    obs_counts = Counter(pairs)
    for a in range(1, 6):
        for m in range(1, 6):
            weight = ((a - m) ** 2) / n_minus_1_sq
            observed_prob = obs_counts[(a, m)] / n_scored
            expected_prob = (auto_counts[a] / n_scored) * (manual_counts[m] / n_scored)
            numer += weight * observed_prob
            denom += weight * expected_prob
    if denom <= 1e-12:
        weighted_kappa = 1.0 if numer <= 1e-12 else 0.0
    else:
        weighted_kappa = 1.0 - (numer / denom)

    return {
        "n_scored": n_scored,
        "agreement_exact": agreement_exact,
        "kappa": float(kappa),
        "kappa_weighted_quadratic": float(weighted_kappa),
    }


def compute_review_agreement(review_rows: list[dict[str, str]]) -> dict[str, float | int]:
    pairs: list[tuple[int, int]] = []
    for row in review_rows:
        auto_txt = str(row.get("automatic_grade", "")).strip()
        manual_txt = str(row.get("manual_grade", "")).strip()
        if not auto_txt or not manual_txt:
            continue
        try:
            auto_grade = _to_int(auto_txt)
            manual_grade = _to_int(manual_txt)
        except Exception:
            continue
        if auto_grade < 1 or auto_grade > 5:
            continue
        if manual_grade < 1 or manual_grade > 5:
            continue
        pairs.append((auto_grade, manual_grade))
    return compute_grade_pair_agreement(pairs)


def _build_auto_review_row(prediction_row: dict[str, str], threshold: int, training_mode: bool) -> dict[str, str]:
    auto_grade = _to_int(prediction_row.get("automatic_grade"))
    auto_conf = _to_int(prediction_row.get("automatic_confidence"))
    now = utc_now_iso()

    if training_mode:
        status = "training_pending"
        final_grade = ""
        reviewed_at = ""
    elif auto_conf >= threshold:
        status = "auto_accepted"
        final_grade = str(auto_grade)
        reviewed_at = now
    else:
        status = "pending"
        final_grade = ""
        reviewed_at = ""

    return {
        "scan_id": prediction_row.get("scan_id", ""),
        "subject_id": prediction_row.get("subject_id", ""),
        "site": prediction_row.get("site", ""),
        "session_id": prediction_row.get("session_id", ""),
        "automatic_grade": str(auto_grade),
        "automatic_confidence": str(auto_conf),
        "confidence_threshold": str(int(threshold)),
        "manual_grade": "",
        "final_grade": final_grade,
        "review_status": status,
        "training_mode": "1" if training_mode else "0",
        "prediction_revealed_at": "",
        "reviewer": "",
        "reviewed_at": reviewed_at,
        "last_updated": now,
    }


def _reset_row_to_unreviewed(row: dict[str, str], now: str) -> None:
    threshold = _to_int(row.get("confidence_threshold"), default=75)
    auto_grade = _to_int(row.get("automatic_grade"), default=0)
    auto_conf = _to_int(row.get("automatic_confidence"), default=0)
    training_mode = _to_bool(row.get("training_mode"), default=False)

    row["manual_grade"] = ""
    row["reviewer"] = ""
    row["reviewed_at"] = ""
    row["prediction_revealed_at"] = ""

    if training_mode:
        row["review_status"] = "training_pending"
        row["final_grade"] = ""
    elif auto_conf >= threshold:
        row["review_status"] = "auto_accepted"
        row["final_grade"] = str(auto_grade)
        row["reviewed_at"] = now
    else:
        row["review_status"] = "pending"
        row["final_grade"] = ""
    row["last_updated"] = now


def initialize_or_update_review(
    review_tsv_path: Path,
    review_json_path: Path,
    review_audit_path: Path,
    prediction_rows: list[dict[str, str]],
    confidence_threshold: int,
    training_mode: bool = False,
) -> list[dict[str, str]]:
    existing_rows = {row["scan_id"]: row for row in read_tsv(review_tsv_path) if row.get("scan_id")}
    audit_rows = read_tsv(review_audit_path)

    out_rows: list[dict[str, str]] = []
    now = utc_now_iso()
    for pred in prediction_rows:
        scan_id = pred.get("scan_id", "")
        if not scan_id:
            continue

        auto_row = _build_auto_review_row(pred, confidence_threshold, bool(training_mode))
        existing = existing_rows.get(scan_id)
        if existing:
            status = existing.get("review_status", "")
            if status.startswith("manual"):
                existing["confidence_threshold"] = str(int(confidence_threshold))
                existing["automatic_confidence"] = auto_row["automatic_confidence"]
                existing["automatic_grade"] = auto_row["automatic_grade"]
                existing["training_mode"] = "1" if training_mode else "0"
                existing["last_updated"] = now
                out_rows.append(existing)
                continue

            if status.startswith("training") and str(existing.get("manual_grade", "")).strip():
                existing["confidence_threshold"] = str(int(confidence_threshold))
                existing["automatic_confidence"] = auto_row["automatic_confidence"]
                existing["automatic_grade"] = auto_row["automatic_grade"]
                existing["training_mode"] = "1" if training_mode else "0"
                existing["last_updated"] = now
                out_rows.append(existing)
                continue

            # Re-evaluate auto state for threshold changes.
            existing.update(auto_row)
            out_rows.append(existing)
            continue

        out_rows.append(auto_row)
        audit_rows.append(
            {
                "timestamp": now,
                "scan_id": scan_id,
                "event": "init",
                "automatic_grade": auto_row["automatic_grade"],
                "manual_grade": "",
                "final_grade": auto_row["final_grade"],
                "reviewer": "",
                "notes": f"confidence_threshold={confidence_threshold};training_mode={int(bool(training_mode))}",
            }
        )

    out_rows = sorted(out_rows, key=lambda row: row["scan_id"])
    write_tsv(review_tsv_path, out_rows, REVIEW_FIELDS)
    write_tsv(review_audit_path, audit_rows, AUDIT_FIELDS)
    _write_review_summary_json(
        review_json_path=review_json_path,
        rows=out_rows,
        confidence_threshold=int(confidence_threshold),
        training_mode=bool(training_mode),
        updated_at=now,
    )
    return out_rows


def apply_manual_review(
    review_tsv_path: Path,
    review_audit_path: Path,
    scan_id: str,
    manual_grade: int,
    reviewer: str,
    review_json_path: Path | None = None,
    training_mode: bool | None = None,
) -> dict[str, str]:
    rows = read_tsv(review_tsv_path)
    if not rows:
        raise FileNotFoundError(f"Review table is missing or empty: {review_tsv_path}")

    hit: dict[str, str] | None = None
    now = utc_now_iso()
    for row in rows:
        if row.get("scan_id") != scan_id:
            continue
        auto_grade = _to_int(row.get("automatic_grade"))
        row_training_mode = _to_bool(row.get("training_mode"), default=False)
        if training_mode is not None:
            row_training_mode = bool(training_mode)
        row["manual_grade"] = str(int(manual_grade))
        row["final_grade"] = str(int(manual_grade))
        if row_training_mode:
            row["review_status"] = "training_completed"
            row["prediction_revealed_at"] = now
        else:
            row["review_status"] = "manual_confirmed" if int(manual_grade) == auto_grade else "manual_override"
            row["prediction_revealed_at"] = row.get("prediction_revealed_at", "")
        row["training_mode"] = "1" if row_training_mode else "0"
        row["reviewer"] = reviewer
        row["reviewed_at"] = now
        row["last_updated"] = now
        hit = row
        break

    if hit is None:
        raise KeyError(f"scan_id not found in review table: {scan_id}")

    write_tsv(review_tsv_path, rows, REVIEW_FIELDS)

    audit_rows = read_tsv(review_audit_path)
    audit_rows.append(
        {
            "timestamp": now,
            "scan_id": scan_id,
            "event": hit["review_status"],
            "automatic_grade": hit.get("automatic_grade", ""),
            "manual_grade": hit.get("manual_grade", ""),
            "final_grade": hit.get("final_grade", ""),
            "reviewer": reviewer,
            "notes": "",
        }
    )
    write_tsv(review_audit_path, audit_rows, AUDIT_FIELDS)

    threshold, training_json = _read_review_json_settings(review_json_path)
    _write_review_summary_json(
        review_json_path=review_json_path,
        rows=rows,
        confidence_threshold=int(threshold),
        training_mode=bool(training_mode) if training_mode is not None else bool(training_json),
        updated_at=now,
    )

    return hit


def clear_manual_reviews(
    review_tsv_path: Path,
    review_audit_path: Path,
    review_json_path: Path | None = None,
    reviewer: str | None = None,
    scan_id: str | None = None,
) -> int:
    rows = read_tsv(review_tsv_path)
    if not rows:
        raise FileNotFoundError(f"Review table is missing or empty: {review_tsv_path}")

    reviewer_filter = str(reviewer).strip() if reviewer is not None else ""
    scan_filter = str(scan_id).strip() if scan_id is not None else ""
    now = utc_now_iso()

    cleared = 0
    scan_id_found = False if scan_filter else True
    audit_rows = read_tsv(review_audit_path)
    for row in rows:
        rid = str(row.get("scan_id", "")).strip()
        if scan_filter and rid != scan_filter:
            continue
        if scan_filter and rid == scan_filter:
            scan_id_found = True

        row_reviewer = str(row.get("reviewer", "")).strip()
        if reviewer_filter and row_reviewer != reviewer_filter:
            continue

        manual_grade = str(row.get("manual_grade", "")).strip()
        status = str(row.get("review_status", "")).strip()
        has_manual = bool(manual_grade) or status in {"manual_confirmed", "manual_override", "training_completed"}
        if not has_manual:
            continue

        old_reviewer = row_reviewer
        _reset_row_to_unreviewed(row, now)
        cleared += 1

        notes = []
        if reviewer_filter:
            notes.append(f"reviewer_filter={reviewer_filter}")
        if scan_filter:
            notes.append(f"scan_filter={scan_filter}")
        audit_rows.append(
            {
                "timestamp": now,
                "scan_id": rid,
                "event": "clear_manual",
                "automatic_grade": row.get("automatic_grade", ""),
                "manual_grade": "",
                "final_grade": row.get("final_grade", ""),
                "reviewer": old_reviewer,
                "notes": ";".join(notes),
            }
        )

    if not scan_id_found:
        raise KeyError(f"scan_id not found in review table: {scan_filter}")

    if cleared > 0:
        write_tsv(review_tsv_path, rows, REVIEW_FIELDS)
        write_tsv(review_audit_path, audit_rows, AUDIT_FIELDS)
        threshold, training_mode = _read_review_json_settings(review_json_path)
        _write_review_summary_json(
            review_json_path=review_json_path,
            rows=rows,
            confidence_threshold=int(threshold),
            training_mode=bool(training_mode),
            updated_at=now,
        )

    return int(cleared)


def _manual_grade_or_none(text: str | int | None) -> int | None:
    try:
        value = _to_int(text, default=0)
    except Exception:
        return None
    if 1 <= value <= 5:
        return int(value)
    return None


def _reviewer_grade_map_for_scan(
    review_row: dict[str, str],
    audit_rows_for_scan: list[dict[str, str]],
) -> dict[str, int]:
    reviewer_grades: dict[str, int] = {}

    for audit_row in sorted(audit_rows_for_scan, key=lambda row: str(row.get("timestamp", ""))):
        reviewer = str(audit_row.get("reviewer", "")).strip()
        if not reviewer:
            continue

        event = str(audit_row.get("event", "")).strip()
        if event == "clear_manual":
            reviewer_grades.pop(reviewer, None)
            continue

        manual_grade = _manual_grade_or_none(audit_row.get("manual_grade"))
        if manual_grade is None:
            continue
        reviewer_grades[reviewer] = manual_grade

    # Keep current review table state authoritative if present.
    reviewer_current = str(review_row.get("reviewer", "")).strip()
    manual_current = _manual_grade_or_none(review_row.get("manual_grade"))
    if reviewer_current and manual_current is not None:
        reviewer_grades[reviewer_current] = manual_current

    return reviewer_grades


def export_reviews(index_rows: list[dict[str, str]], derivatives_root: Path, output_path: Path) -> Path:
    export_rows: list[dict[str, str]] = []
    all_reviewers: set[str] = set()

    for row in index_rows:
        review_rel = row.get("review_tsv", "")
        if not review_rel:
            continue
        review_path = (derivatives_root / review_rel).resolve()

        audit_by_scan: dict[str, list[dict[str, str]]] = {}
        audit_rel = row.get("review_audit", "")
        if audit_rel:
            audit_path = (derivatives_root / audit_rel).resolve()
            if audit_path.exists():
                for audit_row in read_tsv(audit_path):
                    scan_id = str(audit_row.get("scan_id", "")).strip()
                    if not scan_id:
                        continue
                    audit_by_scan.setdefault(scan_id, []).append(audit_row)

        for review_row in read_tsv(review_path):
            scan_id = str(review_row.get("scan_id", "")).strip()
            reviewer_grades = _reviewer_grade_map_for_scan(
                review_row=review_row,
                audit_rows_for_scan=audit_by_scan.get(scan_id, []),
            )
            reviewers_for_scan = sorted(reviewer_grades.keys(), key=str.casefold)
            all_reviewers.update(reviewers_for_scan)

            grade_values = [reviewer_grades[reviewer] for reviewer in reviewers_for_scan]
            if grade_values:
                consensus_mean = sum(float(v) for v in grade_values) / float(len(grade_values))
                consensus_mean_txt = f"{consensus_mean:.3f}"
                consensus_rounded = int(min(5, max(1, int(consensus_mean + 0.5))))
                consensus_rounded_txt = str(consensus_rounded)
            else:
                consensus_mean_txt = ""
                consensus_rounded_txt = ""

            export_rows.append(
                {
                    "scan_id": scan_id,
                    "subject_id": review_row.get("subject_id", ""),
                    "site": review_row.get("site", ""),
                    "session_id": review_row.get("session_id", ""),
                    "automatic_grade": review_row.get("automatic_grade", ""),
                    "automatic_confidence": review_row.get("automatic_confidence", ""),
                    "manual_grade": review_row.get("manual_grade", ""),
                    "final_grade": review_row.get("final_grade", ""),
                    "review_status": review_row.get("review_status", ""),
                    "reviewer": review_row.get("reviewer", ""),
                    "reviewed_at": review_row.get("reviewed_at", ""),
                    "raw_image_path": row.get("raw_image_path", ""),
                    "attention_map_path": row.get("attention_map_path", ""),
                    "reviewer_count": str(len(reviewers_for_scan)),
                    "reviewers": "|".join(reviewers_for_scan),
                    "consensus_method": "mean_manual_grade" if grade_values else "",
                    "consensus_mean_manual_grade": consensus_mean_txt,
                    "consensus_grade_rounded": consensus_rounded_txt,
                    "__reviewer_grades": reviewer_grades,
                }
            )

    reviewer_order = sorted(all_reviewers, key=str.casefold)
    for export_row in export_rows:
        reviewer_grades = export_row.pop("__reviewer_grades", {})
        for idx, reviewer in enumerate(reviewer_order, start=1):
            grade = reviewer_grades.get(reviewer)
            export_row[f"reviewer_{idx}_id"] = reviewer
            export_row[f"reviewer_{idx}_grade"] = str(grade) if grade is not None else ""

    export_rows = sorted(export_rows, key=lambda r: r["scan_id"])
    fields = [
        "scan_id",
        "subject_id",
        "site",
        "session_id",
        "automatic_grade",
        "automatic_confidence",
        "manual_grade",
        "final_grade",
        "review_status",
        "reviewer",
        "reviewed_at",
        "raw_image_path",
        "attention_map_path",
        "reviewer_count",
        "reviewers",
        "consensus_method",
        "consensus_mean_manual_grade",
        "consensus_grade_rounded",
    ]
    for idx, _reviewer in enumerate(reviewer_order, start=1):
        fields.append(f"reviewer_{idx}_id")
        fields.append(f"reviewer_{idx}_grade")
    write_tsv(output_path, export_rows, fields)
    return output_path
