from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.filters import gaussian

from motionscore.inference.preprocessing import preprocess_slice
from motionscore.io.aim import read_aim
from motionscore.utils import read_tsv, to_relpath, write_tsv

TRAIN_MANIFEST_FIELDS = [
    "scan_id",
    "subject_id",
    "raw_image_path",
    "slice_index",
    "label",
    "label_source",
    "sample_weight",
    "split",
    "is_manual",
    "manual_grade",
    "automatic_grade",
    "automatic_confidence",
    "auto_slice_confidence",
    "cache_npy_path",
    "cache_index",
    "fold_id",
]

AUTO_CONFIDENCE_EPSILON = 1e-6


def _to_int(text: str | int | float | None, default: int = 0) -> int:
    if text is None:
        return default
    if isinstance(text, int):
        return text
    text_str = str(text).strip()
    if not text_str:
        return default
    return int(float(text_str))


def _parse_json_list(text: str | None) -> list:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def _confidence_as_unit_interval(value: float | int | str | None) -> float:
    try:
        conf = float(value)
    except Exception:
        return -1.0
    if conf > 1.0:
        conf = conf / 100.0
    return float(conf)


def _subject_split(subject_id: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    key = f"{seed}:{subject_id}".encode("utf-8")
    u = int(hashlib.sha1(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    if u < train_ratio:
        return "train"
    if u < (train_ratio + val_ratio):
        return "val"
    return "test"


def _assign_subject_splits(
    subject_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    subjects = [str(sid).strip() for sid in subject_ids if str(sid).strip()]
    n = len(subjects)
    if n <= 0:
        return {}
    if n == 1:
        return {subjects[0]: "train"}
    if n == 2:
        ranked = sorted(
            subjects,
            key=lambda sid: hashlib.sha1(f"{seed}:{sid}".encode("utf-8")).hexdigest(),
        )
        return {ranked[0]: "train", ranked[1]: "test"}

    ranked = sorted(
        subjects,
        key=lambda sid: hashlib.sha1(f"{seed}:{sid}".encode("utf-8")).hexdigest(),
    )
    desired = [float(train_ratio) * n, float(val_ratio) * n, float(test_ratio) * n]
    counts = [int(np.floor(value)) for value in desired]
    remainder = int(n - sum(counts))
    fractions = [desired[i] - counts[i] for i in range(3)]
    for idx in sorted(range(3), key=lambda i: fractions[i], reverse=True)[:remainder]:
        counts[idx] += 1

    for idx in range(3):
        if counts[idx] > 0:
            continue
        donor = max(range(3), key=lambda j: counts[j])
        if counts[donor] <= 1:
            continue
        counts[donor] -= 1
        counts[idx] += 1

    while sum(counts) > n:
        donor = max(range(3), key=lambda j: counts[j])
        if counts[donor] <= 1:
            break
        counts[donor] -= 1
    while sum(counts) < n:
        receiver = max(range(3), key=lambda j: desired[j] - counts[j])
        counts[receiver] += 1

    labels = ["train", "val", "test"]
    out: dict[str, str] = {}
    offset = 0
    for label, count in zip(labels, counts):
        for sid in ranked[offset:offset + count]:
            out[sid] = label
        offset += count
    return out


def _assign_subject_folds(subject_ids: list[str], *, seed: int, cv_folds: int) -> dict[str, int]:
    n_folds = int(max(2, cv_folds))
    ranked = sorted(
        [str(sid).strip() for sid in subject_ids if str(sid).strip()],
        key=lambda sid: hashlib.sha1(f"{seed}:{sid}".encode("utf-8")).hexdigest(),
    )
    out: dict[str, int] = {}
    for idx, sid in enumerate(ranked):
        out[sid] = int(idx % n_folds)
    return out


def _slice_indices_for_scan(
    depth: int,
    *,
    slice_step: int,
    slice_count: int,
    seed: int,
    scan_id: str,
) -> list[int]:
    n = int(max(0, depth))
    if n <= 0:
        return []
    count = int(max(0, slice_count))
    if count > 0:
        k = int(min(n, count))
        key = f"{int(seed)}:{str(scan_id).strip()}".encode("utf-8")
        seed_u32 = int(hashlib.sha1(key).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_u32)
        edges = np.linspace(0, n, num=(k + 1), dtype=np.int64)
        out: list[int] = []
        for i in range(k):
            lo = int(edges[i])
            hi = int(edges[i + 1])
            if hi <= lo:
                hi = lo + 1
            z = int(rng.integers(lo, hi))
            z = int(max(0, min(n - 1, z)))
            out.append(z)
        return sorted(set(out))

    step = int(max(1, slice_step))
    return list(range(0, n, step))


def _sample_candidate_positions(
    n_candidates: int,
    *,
    slice_count: int,
    seed: int,
    scan_id: str,
) -> list[int]:
    n = int(max(0, n_candidates))
    if n <= 0:
        return []
    count = int(max(0, slice_count))
    if count <= 0:
        return list(range(n))
    k = int(min(n, count))
    key = f"{int(seed)}:{str(scan_id).strip()}".encode("utf-8")
    seed_u32 = int(hashlib.sha1(key).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_u32)
    edges = np.linspace(0, n, num=(k + 1), dtype=np.int64)
    out: list[int] = []
    for i in range(k):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi <= lo:
            hi = lo + 1
        pos = int(rng.integers(lo, hi))
        pos = int(max(0, min(n - 1, pos)))
        out.append(pos)
    return sorted(set(out))


def _infer_depth_from_raw(raw_image_path: Path, scaling: str = "native") -> int:
    aim = read_aim(raw_image_path, scaling=scaling)
    if aim.data.ndim != 3:
        raise ValueError(f"Expected 3D AIM volume for {raw_image_path}, got shape {aim.data.shape}")
    return int(aim.data.shape[2])


def _build_slice_cache_db(
    rows: list[dict[str, str]],
    *,
    output_manifest_path: Path,
    slice_db_dir: Path,
    scaling: str = "native",
) -> tuple[list[dict[str, str]], dict[str, int]]:
    slice_db_dir = slice_db_dir.resolve()
    slice_db_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        raw = str(row.get("raw_image_path", "")).strip()
        if not raw:
            continue
        grouped_rows.setdefault(raw, []).append(row)

    written_scans = 0
    written_slices = 0
    manifest_parent = output_manifest_path.resolve().parent

    for raw_path_txt, scan_rows in grouped_rows.items():
        raw_path = Path(raw_path_txt).resolve()
        z_list = sorted(
            {
                int(str(r.get("slice_index", "0")).strip() or "0")
                for r in scan_rows
            }
        )
        if not z_list:
            continue

        aim = read_aim(raw_path, scaling=scaling)
        if aim.data.ndim != 3:
            raise ValueError(f"Expected 3D AIM volume for {raw_path}, got shape {aim.data.shape}")
        filtered = gaussian(aim.data, sigma=0.8, truncate=1.25)

        safe_z = [int(max(0, min(filtered.shape[2] - 1, z))) for z in z_list]
        images: list[np.ndarray] = []
        z_to_local: dict[int, int] = {}
        for idx, z in enumerate(safe_z):
            arr, _ = preprocess_slice(filtered[:, :, z])
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[2] > 0:
                arr = arr[:, :, 0]
            images.append(arr.astype(np.uint8))
            z_to_local[z] = idx

        cache_key = hashlib.sha1(raw_path.as_posix().encode("utf-8")).hexdigest()[:16]
        cache_path = slice_db_dir / f"{cache_key}.npy"
        if not images:
            continue
        stacked = np.stack(images, axis=0)
        np.save(cache_path, stacked, allow_pickle=False)
        rel_cache = to_relpath(cache_path, manifest_parent)

        for row in scan_rows:
            z = int(str(row.get("slice_index", "0")).strip() or "0")
            z = int(max(0, min(filtered.shape[2] - 1, z)))
            row["cache_npy_path"] = rel_cache
            row["cache_index"] = str(int(z_to_local[z]))
            written_slices += 1
        written_scans += 1

    return rows, {"cache_scans": written_scans, "cache_slices": written_slices}


def build_training_manifest(
    derivatives_root: Path,
    output_path: Path,
    *,
    min_auto_confidence: float,
    slice_step: int,
    include_auto_without_manual: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    slice_count: int = 0,
    cv_folds: int = 10,
    scaling: str = "native",
) -> dict[str, int | float]:
    step = int(slice_step)
    if step <= 0:
        raise ValueError("slice_step must be > 0")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("train/val/test ratios must be > 0")
    ratio_sum = float(train_ratio + val_ratio + test_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if int(cv_folds) < 2:
        raise ValueError("cv_folds must be >= 2")

    index_rows = read_tsv(derivatives_root / "index.tsv")
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    subject_ids = sorted({str(row.get("subject_id", "")).strip() or str(row.get("scan_id", "")).strip() for row in index_rows})
    split_by_subject = _assign_subject_splits(
        subject_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    fold_by_subject = _assign_subject_folds(subject_ids, seed=seed, cv_folds=int(cv_folds))

    out_rows: list[dict[str, str]] = []
    manual_rows = 0
    auto_rows = 0
    skipped_scans = 0

    for index_row in index_rows:
        scan_id = str(index_row.get("scan_id", "")).strip()
        subject_id = str(index_row.get("subject_id", "")).strip() or scan_id
        raw_image_path_txt = str(index_row.get("raw_image_path", "")).strip()
        if not scan_id or not raw_image_path_txt:
            skipped_scans += 1
            continue

        raw_image_path = Path(raw_image_path_txt).resolve()
        split = split_by_subject.get(subject_id, "train")
        fold_id = int(fold_by_subject.get(subject_id, 0))

        predictions_rel = str(index_row.get("predictions_tsv", "")).strip()
        review_rel = str(index_row.get("review_tsv", "")).strip()

        pred_row = {}
        if predictions_rel:
            pred_rows = read_tsv((derivatives_root / predictions_rel).resolve())
            if pred_rows:
                pred_row = pred_rows[0]

        review_row = {}
        if review_rel:
            review_rows = read_tsv((derivatives_root / review_rel).resolve())
            if review_rows:
                review_row = review_rows[0]

        manual_grade = _to_int(review_row.get("manual_grade"), default=0)
        auto_grade = _to_int(pred_row.get("automatic_grade"), default=0)
        auto_conf = _to_int(pred_row.get("automatic_confidence"), default=0)
        slice_grades = _parse_json_list(pred_row.get("slice_grades"))
        slice_conf = _parse_json_list(pred_row.get("slice_confidences"))

        if 1 <= manual_grade <= 5:
            depth = len(slice_grades)
            if depth <= 0:
                depth = _infer_depth_from_raw(raw_image_path, scaling=scaling)
            sample_indices = _slice_indices_for_scan(
                int(depth),
                slice_step=step,
                slice_count=int(slice_count),
                seed=int(seed),
                scan_id=scan_id,
            )
            for z in sample_indices:
                out_rows.append(
                    {
                        "scan_id": scan_id,
                        "subject_id": subject_id,
                        "raw_image_path": str(raw_image_path),
                        "slice_index": str(int(z)),
                        "label": str(int(manual_grade)),
                        "label_source": "manual_scan_propagated",
                        "sample_weight": "1.000",
                        "split": split,
                        "is_manual": "1",
                        "manual_grade": str(int(manual_grade)),
                        "automatic_grade": str(int(auto_grade)) if 1 <= auto_grade <= 5 else "",
                        "automatic_confidence": str(int(auto_conf)) if auto_conf >= 0 else "",
                        "auto_slice_confidence": "",
                        "fold_id": str(int(fold_id)),
                    }
                )
                manual_rows += 1
            continue

        if not include_auto_without_manual:
            skipped_scans += 1
            continue

        if not slice_grades:
            skipped_scans += 1
            continue

        n = len(slice_grades)
        eligible_auto: list[tuple[int, int, float]] = []
        for z in range(n):
            label = _to_int(slice_grades[z], default=0)
            if label < 1 or label > 5:
                continue
            conf_unit = _confidence_as_unit_interval(slice_conf[z] if z < len(slice_conf) else -1)
            if not np.isfinite(conf_unit) or conf_unit < 0.0:
                continue
            if conf_unit + AUTO_CONFIDENCE_EPSILON < float(min_auto_confidence):
                continue
            eligible_auto.append((z, label, float(conf_unit)))

        if not eligible_auto:
            skipped_scans += 1
            continue

        if int(slice_count) > 0:
            candidate_positions = _sample_candidate_positions(
                len(eligible_auto),
                slice_count=int(slice_count),
                seed=int(seed),
                scan_id=scan_id,
            )
            chosen_auto = [eligible_auto[pos] for pos in candidate_positions]
        else:
            chosen_auto = eligible_auto[::step]

        for z, label, conf_unit in chosen_auto:
            out_rows.append(
                {
                    "scan_id": scan_id,
                    "subject_id": subject_id,
                    "raw_image_path": str(raw_image_path),
                    "slice_index": str(int(z)),
                    "label": str(int(label)),
                    "label_source": "auto_slice",
                    "sample_weight": f"{float(max(0.05, conf_unit)):.3f}",
                    "split": split,
                    "is_manual": "0",
                    "manual_grade": "",
                    "automatic_grade": str(int(auto_grade)) if 1 <= auto_grade <= 5 else "",
                    "automatic_confidence": str(int(auto_conf)) if auto_conf >= 0 else "",
                    "auto_slice_confidence": f"{float(conf_unit):.4f}",
                    "cache_npy_path": "",
                    "cache_index": "",
                    "fold_id": str(int(fold_id)),
                }
            )
            auto_rows += 1

    out_rows = sorted(out_rows, key=lambda r: (r["split"], r["scan_id"], int(r["slice_index"])))
    cache_stats = {"cache_scans": 0, "cache_slices": 0}
    if out_rows:
        target_db_dir = output_path.resolve().parent / "slice_db"
        out_rows, cache_stats = _build_slice_cache_db(
            out_rows,
            output_manifest_path=output_path,
            slice_db_dir=target_db_dir,
            scaling=scaling,
        )
    write_tsv(output_path, out_rows, TRAIN_MANIFEST_FIELDS)

    split_counts = {"train": 0, "val": 0, "test": 0}
    fold_counts: dict[int, int] = {}
    for row in out_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        fold = _to_int(row.get("fold_id"), default=0)
        fold_counts[fold] = int(fold_counts.get(fold, 0) + 1)

    return {
        "rows_written": len(out_rows),
        "rows_manual": manual_rows,
        "rows_auto": auto_rows,
        "scans_seen": len(index_rows),
        "scans_skipped": skipped_scans,
        "split_train": split_counts.get("train", 0),
        "split_val": split_counts.get("val", 0),
        "split_test": split_counts.get("test", 0),
        "cv_folds": int(cv_folds),
        "folds_populated": int(len([1 for v in fold_counts.values() if int(v) > 0])),
        "cache_scans": cache_stats.get("cache_scans", 0),
        "cache_slices": cache_stats.get("cache_slices", 0),
    }
