from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

from motionscore import __version__
from motionscore.config import AppConfig
from motionscore.dataset.discovery import discover_raw_sessions
from motionscore.dataset.layout import (
    PIPELINE_NAME,
    get_dataset_description_path,
    get_derivatives_root,
    get_explain_dir,
    get_export_path,
    get_index_path,
    get_preview_dir,
    get_predictions_dir,
    get_review_dir,
)
from motionscore.dataset.models import RawSession
from motionscore.inference.scoring import predict_scan
from motionscore.io.aim import read_aim
from motionscore.model_registry import (
    get_registry_path,
    list_model_profiles,
    register_model_profile,
    resolve_model_dir,
)
from motionscore.review.store import (
    apply_manual_review,
    clear_manual_reviews,
    export_reviews,
    import_final_grades,
    initialize_or_update_review,
)
from motionscore.review.preview import write_prediction_preview_png, write_slice_profile_png
from motionscore.training.prepare import build_training_manifest
from motionscore.training.trainer import TrainConfig, run_transfer_learning
from motionscore.utils import (
    make_scan_id,
    read_tsv,
    to_relpath,
    utc_now_iso,
    write_json,
    write_tsv,
)

INDEX_FIELDS = [
    "scan_id",
    "subject_id",
    "raw_image_path",
    "predictions_tsv",
    "review_tsv",
    "review_json",
    "review_audit",
    "preview_png_path",
    "slice_profile_png_path",
    "attention_map_path",
    "automatic_grade",
    "automatic_confidence",
    "manual_mode",
    "model_id",
    "model_version",
    "predicted_at",
]

PREDICTION_FIELDS = [
    "scan_id",
    "subject_id",
    "raw_image_path",
    "preview_png_path",
    "slice_profile_png_path",
    "automatic_grade",
    "automatic_confidence",
    "manual_mode",
    "model_id",
    "mean_confidence",
    "stack_ranges",
    "stack_grades",
    "stack_confidences",
    "slice_grades",
    "slice_confidences",
    "model_version",
    "predicted_at",
]


def _print_citation_notice() -> None:
    print("=" * 80)
    print("If you use this tool, please cite:")
    print()
    print("   Walle, M. et al., Bone 166 (2023): 116607")
    print("   https://doi.org/10.1016/j.bone.2022.116607")
    print("=" * 80)
    print()


def _normalize_derivatives_root(path: Path) -> Path:
    if path.name == PIPELINE_NAME:
        return path
    candidate = path / "derivatives" / PIPELINE_NAME
    if candidate.exists():
        return candidate
    candidate2 = path / PIPELINE_NAME
    if candidate2.exists() and path.name == "derivatives":
        return candidate2
    return path


def _read_index(derivatives_root: Path) -> list[dict[str, str]]:
    return read_tsv(get_index_path(derivatives_root))


def _upsert_index_rows(derivatives_root: Path, rows: list[dict[str, str]]) -> None:
    index_path = get_index_path(derivatives_root)
    existing = {row["scan_id"]: row for row in read_tsv(index_path) if row.get("scan_id")}
    for row in rows:
        scan_id = row["scan_id"]
        prev = existing.get(scan_id, {})
        merged = dict(prev)
        merged.update(row)
        if not merged.get("attention_map_path"):
            merged["attention_map_path"] = prev.get("attention_map_path", "")
        existing[scan_id] = merged

    out = sorted(existing.values(), key=lambda r: r["scan_id"])
    write_tsv(index_path, out, INDEX_FIELDS)


def _write_dataset_description(derivatives_root: Path) -> None:
    path = get_dataset_description_path(derivatives_root)
    payload = {
        "Name": PIPELINE_NAME,
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "motionscore",
                "Version": __version__,
                "Description": "MotionScore CNN grading with review and Grad-CAM outputs",
            }
        ],
        "HowToAcknowledge": "Please cite Walle et al. Bone (2023) https://doi.org/10.1016/j.bone.2022.116607",
    }
    write_json(path, payload)


def _session_from_index(row: dict[str, str]) -> RawSession:
    return RawSession(
        subject_id=row.get("subject_id", ""),
        site=row.get("site", "tibia") or "tibia",
        session_id=row.get("session_id", "T1") or "T1",
        raw_image_path=Path(row["raw_image_path"]),
        stack_index=int(row["stack_index"]) if row.get("stack_index") else None,
    )


def _scan_id_for_session(session: RawSession) -> str:
    return make_scan_id(session.subject_id, session.site, session.session_id, session.raw_image_path)


def _default_model_root() -> Path:
    return (Path.home() / ".motionscore" / "MotionScore" / "models").resolve()


def _resolve_model_root(model_root: Path | None) -> Path:
    return model_root.resolve() if model_root is not None else _default_model_root()


def _sanitize_model_component(model_id: str | None) -> str:
    raw = str(model_id or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")
    return safe or "default"


def _requested_model_storage_id(args: argparse.Namespace) -> str:
    model_dir = getattr(args, "model_dir", None)
    if model_dir is not None:
        try:
            return Path(model_dir).resolve().name
        except Exception:
            return Path(model_dir).name
    model_id = str(getattr(args, "model_id", "") or "").strip()
    return model_id or "base-v1"


def _session_output_paths(
    *,
    derivatives_root: Path,
    session: RawSession,
    model_id: str,
    scan_id: str,
) -> dict[str, Path]:
    pred_dir = get_predictions_dir(derivatives_root, session)
    review_dir = get_review_dir(derivatives_root, session)
    preview_dir = get_preview_dir(derivatives_root, session)
    model_component = _sanitize_model_component(model_id)
    model_pred_dir = pred_dir / "models" / model_component
    model_review_dir = review_dir / "models" / model_component
    model_preview_dir = preview_dir / "models" / model_component
    return {
        "predictions_tsv": model_pred_dir / "predictions.tsv",
        "review_tsv": model_review_dir / "review.tsv",
        "review_json": model_review_dir / "review.json",
        "review_audit": model_review_dir / "review_audit.tsv",
        "preview_png": model_preview_dir / f"{scan_id}_preview.png",
        "slice_profile_png": model_preview_dir / f"{scan_id}_slice_profile.png",
    }


def _legacy_session_output_paths(*, derivatives_root: Path, session: RawSession, scan_id: str) -> dict[str, Path]:
    pred_dir = get_predictions_dir(derivatives_root, session)
    review_dir = get_review_dir(derivatives_root, session)
    preview_dir = get_preview_dir(derivatives_root, session)
    return {
        "predictions_tsv": pred_dir / "predictions.tsv",
        "review_tsv": review_dir / "review.tsv",
        "review_json": review_dir / "review.json",
        "review_audit": review_dir / "review_audit.tsv",
        "preview_png": preview_dir / f"{scan_id}_preview.png",
        "slice_profile_png": preview_dir / f"{scan_id}_slice_profile.png",
    }


def _cmd_discover(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    sessions = discover_raw_sessions(
        root=args.input_root,
        cfg=cfg.discovery,
        force_header_discovery=bool(args.force_header_discovery),
    )
    rows = [
        {
            "subject_id": s.subject_id,
            "site": s.site,
            "session_id": s.session_id,
            "stack_index": s.stack_index,
            "raw_image_path": str(s.raw_image_path),
        }
        for s in sessions
    ]

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"[motionscore] discovered {len(rows)} scan(s)")
    for row in rows:
        stack_txt = f" stack={row['stack_index']}" if row["stack_index"] is not None else ""
        print(
            f"  sub-{row['subject_id']} site-{row['site']} ses-{row['session_id']}{stack_txt}"
            f" -> {row['raw_image_path']}"
        )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from motionscore.inference.model import ModelEnsemble

    cfg = AppConfig()
    sessions = discover_raw_sessions(
        root=args.input_root,
        cfg=cfg.discovery,
        force_header_discovery=bool(args.force_header_discovery),
    )
    if args.scan_id:
        requested = {str(v).strip() for v in args.scan_id if str(v).strip()}
        if not requested:
            raise ValueError("At least one non-empty --scan-id must be provided")
        filtered: list[RawSession] = []
        missing = set(requested)
        for session in sessions:
            sid = _scan_id_for_session(session)
            if sid in requested:
                filtered.append(session)
                missing.discard(sid)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"--scan-id not found in discovered dataset: {missing_list}")
        sessions = filtered

    if not sessions:
        print(f"[motionscore] no AIM scans discovered under {args.input_root}")
        return 0

    derivatives_root = get_derivatives_root(args.input_root, args.output_root).resolve()
    derivatives_root.mkdir(parents=True, exist_ok=True)
    _write_dataset_description(derivatives_root)

    confidence_threshold = int(args.confidence_threshold)
    if confidence_threshold < 0 or confidence_threshold > 100:
        raise ValueError("confidence threshold must be in [0, 100]")
    preview_panels = int(max(1, min(9, int(args.preview_panels))))
    manual_only = bool(getattr(args, "manual_only", False))
    if manual_only and bool(args.save_preview_png):
        print("[predict] manual-only mode: preview PNG export disabled")

    model_root = _resolve_model_root(args.model_root)
    model_id = str(getattr(args, "model_id", "base-v1") or "base-v1").strip()
    requested_storage_model_id = _requested_model_storage_id(args)
    ensemble = None
    model_version = "manual-only" if manual_only else ""
    resolved_model_id = "manual-only" if manual_only else model_id
    if manual_only:
        print("[predict] manual-only mode: skipping CNN inference")
    else:
        ensemble = ModelEnsemble(
            model_dir=args.model_dir,
            model_root=model_root,
            model_id=model_id,
            device=args.device,
        )
        model_version = ensemble.model_identity()
        resolved_model_id = ensemble.resolved_model_id()
        print(f"[predict] using torch device={ensemble.model_device()}")

    index_updates: list[dict[str, str]] = []
    for session in sessions:
        scan_id = _scan_id_for_session(session)
        storage_model_id = requested_storage_model_id if manual_only else resolved_model_id
        scoped_paths = _session_output_paths(
            derivatives_root=derivatives_root,
            session=session,
            model_id=storage_model_id,
            scan_id=scan_id,
        )
        legacy_paths = _legacy_session_output_paths(
            derivatives_root=derivatives_root,
            session=session,
            scan_id=scan_id,
        )

        predictions_tsv = scoped_paths["predictions_tsv"]
        review_tsv = scoped_paths["review_tsv"]
        review_json = scoped_paths["review_json"]
        review_audit = scoped_paths["review_audit"]

        existing_prediction_row = {}
        if manual_only:
            lookup_paths = [predictions_tsv]
            if legacy_paths["predictions_tsv"] != predictions_tsv:
                lookup_paths.append(legacy_paths["predictions_tsv"])
            for candidate_path in lookup_paths:
                if not candidate_path.exists():
                    continue
                existing_rows = read_tsv(candidate_path)
                if not existing_rows:
                    continue
                candidate_row = dict(existing_rows[0])
                candidate_model_id = str(candidate_row.get("model_id", "")).strip()
                if candidate_path == legacy_paths["predictions_tsv"] and candidate_model_id not in {"", storage_model_id}:
                    continue
                existing_prediction_row = candidate_row
                break

        if manual_only:
            prediction = None
            aim_volume = None
        else:
            aim_volume = read_aim(session.raw_image_path, scaling=args.scaling)
            prediction = predict_scan(
                aim_volume.data,
                ensemble=ensemble,
                stackheight=int(args.stackheight),
                on_incomplete_stack=args.on_incomplete_stack,
                slice_batch_size=int(args.slice_batch_size),
                slice_step=int(args.slice_step),
                retain_preprocessed=False,
            )

        predicted_at = (
            str(existing_prediction_row.get("predicted_at", "")).strip()
            if manual_only and existing_prediction_row and str(existing_prediction_row.get("automatic_grade", "")).strip()
            else utc_now_iso()
        )
        preserved_model_id = str(existing_prediction_row.get("model_id", "")).strip()
        preserved_model_version = str(existing_prediction_row.get("model_version", "")).strip()
        preserved_auto_grade = str(existing_prediction_row.get("automatic_grade", "")).strip()
        preserved_auto_conf = str(existing_prediction_row.get("automatic_confidence", "")).strip()
        preserved_preview_path = str(existing_prediction_row.get("preview_png_path", "")).strip()
        preserved_profile_path = str(existing_prediction_row.get("slice_profile_png_path", "")).strip()
        preserved_mean_conf = str(existing_prediction_row.get("mean_confidence", "")).strip()
        preserved_stack_ranges = str(existing_prediction_row.get("stack_ranges", "")).strip()
        preserved_stack_grades = str(existing_prediction_row.get("stack_grades", "")).strip()
        preserved_stack_confidences = str(existing_prediction_row.get("stack_confidences", "")).strip()
        preserved_slice_grades = str(existing_prediction_row.get("slice_grades", "")).strip()
        preserved_slice_confidences = str(existing_prediction_row.get("slice_confidences", "")).strip()

        prediction_row = {
            "scan_id": scan_id,
            "subject_id": session.subject_id,
            "raw_image_path": str(session.raw_image_path.resolve()),
            "preview_png_path": preserved_preview_path if manual_only else "",
            "slice_profile_png_path": preserved_profile_path if manual_only else "",
            "automatic_grade": preserved_auto_grade if manual_only else str(prediction.automatic_grade),
            "automatic_confidence": preserved_auto_conf if manual_only else str(prediction.automatic_confidence),
            "manual_mode": "1" if manual_only else "0",
            "model_id": preserved_model_id if (manual_only and preserved_model_id) else resolved_model_id,
            "mean_confidence": preserved_mean_conf if manual_only else str(prediction.mean_confidence),
            "stack_ranges": preserved_stack_ranges if manual_only else json.dumps(prediction.stack_ranges),
            "stack_grades": preserved_stack_grades if manual_only else json.dumps(prediction.stack_grades),
            "stack_confidences": preserved_stack_confidences if manual_only else json.dumps(prediction.stack_confidences),
            "slice_grades": preserved_slice_grades if manual_only else json.dumps(prediction.slice_grades),
            "slice_confidences": preserved_slice_confidences if manual_only else json.dumps(prediction.slice_confidences),
            "model_version": preserved_model_version if (manual_only and preserved_model_version) else model_version,
            "predicted_at": predicted_at,
        }

        if bool(args.save_preview_png) and not manual_only and aim_volume is not None and prediction is not None:
            written_preview = write_prediction_preview_png(
                volume_xyz=aim_volume.data,
                prediction=prediction,
                output_path=scoped_paths["preview_png"],
                max_panels=preview_panels,
            )
            prediction_row["preview_png_path"] = to_relpath(written_preview, derivatives_root)
            try:
                written_profile = write_slice_profile_png(
                    prediction=prediction,
                    output_path=scoped_paths["slice_profile_png"],
                )
                prediction_row["slice_profile_png_path"] = to_relpath(written_profile, derivatives_root)
            except RuntimeError as exc:
                print(
                    "[predict] skipping slice profile PNG (optional dependency missing): "
                    f"{exc}",
                    file=sys.stderr,
                )

        write_tsv(predictions_tsv, [prediction_row], PREDICTION_FIELDS)

        initialize_or_update_review(
            review_tsv_path=review_tsv,
            review_json_path=review_json,
            review_audit_path=review_audit,
            prediction_rows=[prediction_row],
            confidence_threshold=confidence_threshold,
            training_mode=bool(args.training_mode),
        )

        index_updates.append(
            {
                "scan_id": scan_id,
                "subject_id": session.subject_id,
                "raw_image_path": str(session.raw_image_path.resolve()),
                "predictions_tsv": to_relpath(predictions_tsv, derivatives_root),
                "review_tsv": to_relpath(review_tsv, derivatives_root),
                "review_json": to_relpath(review_json, derivatives_root),
                "review_audit": to_relpath(review_audit, derivatives_root),
                "preview_png_path": prediction_row.get("preview_png_path", ""),
                "slice_profile_png_path": prediction_row.get("slice_profile_png_path", ""),
                "attention_map_path": "",
                "automatic_grade": prediction_row["automatic_grade"],
                "automatic_confidence": prediction_row["automatic_confidence"],
                "manual_mode": prediction_row["manual_mode"],
                "model_id": resolved_model_id,
                "model_version": model_version,
                "predicted_at": predicted_at,
            }
        )

        if manual_only:
            if preserved_auto_grade:
                print(f"[predict] {scan_id} manual-only initialized (kept existing AI outputs)")
            else:
                print(f"[predict] {scan_id} manual-only initialized")
        else:
            print(
                f"[predict] {scan_id} grade={prediction.automatic_grade} conf={prediction.automatic_confidence}%"
            )

        # Keep memory bounded across large datasets by releasing per-scan arrays early.
        if prediction is not None:
            prediction.preprocessed_scan = None
        aim_volume = None
        gc.collect()

    _upsert_index_rows(derivatives_root, index_updates)
    print(f"[motionscore] wrote derivatives under: {derivatives_root}")
    return 0


def _cmd_review_init(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    threshold = int(args.confidence_threshold)
    if threshold < 0 or threshold > 100:
        raise ValueError("confidence threshold must be in [0, 100]")

    for row in index_rows:
        prediction_row = {
            "scan_id": row.get("scan_id", ""),
            "subject_id": row.get("subject_id", ""),
            "automatic_grade": row.get("automatic_grade", ""),
            "automatic_confidence": row.get("automatic_confidence", ""),
            "manual_mode": row.get("manual_mode", ""),
        }
        initialize_or_update_review(
            review_tsv_path=derivatives_root / row["review_tsv"],
            review_json_path=derivatives_root / row["review_json"],
            review_audit_path=derivatives_root / row["review_audit"],
            prediction_rows=[prediction_row],
            confidence_threshold=threshold,
            training_mode=bool(args.training_mode),
        )

    print(f"[motionscore] review tables initialized/updated for {len(index_rows)} scan(s)")
    return 0


def _cmd_review_apply(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    hit = next((row for row in index_rows if row.get("scan_id") == args.scan_id), None)
    if hit is None:
        raise KeyError(f"scan_id not found in index.tsv: {args.scan_id}")

    updated = apply_manual_review(
        review_tsv_path=derivatives_root / hit["review_tsv"],
        review_audit_path=derivatives_root / hit["review_audit"],
        scan_id=args.scan_id,
        manual_grade=int(args.manual_grade),
        reviewer=args.reviewer,
        review_json_path=(derivatives_root / hit["review_json"]) if hit.get("review_json") else None,
    )
    print(
        f"[review-apply] {args.scan_id} auto={updated.get('automatic_grade', '')} "
        f"manual={updated.get('manual_grade', '')} final_grade={updated['final_grade']} "
        f"status={updated['review_status']}"
    )
    return 0


def _cmd_review_clear(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    reviewer = str(args.reviewer).strip() if args.reviewer else ""
    if not reviewer and not bool(args.all_reviewers):
        raise ValueError("Specify --reviewer <id> or --all-reviewers")

    total_cleared = 0
    if args.scan_id:
        hit = next((row for row in index_rows if row.get("scan_id") == args.scan_id), None)
        if hit is None:
            raise KeyError(f"scan_id not found in index.tsv: {args.scan_id}")
        total_cleared += clear_manual_reviews(
            review_tsv_path=derivatives_root / hit["review_tsv"],
            review_audit_path=derivatives_root / hit["review_audit"],
            review_json_path=(derivatives_root / hit["review_json"]) if hit.get("review_json") else None,
            reviewer=reviewer if reviewer else None,
            scan_id=args.scan_id,
        )
    else:
        for row in index_rows:
            total_cleared += clear_manual_reviews(
                review_tsv_path=derivatives_root / row["review_tsv"],
                review_audit_path=derivatives_root / row["review_audit"],
                review_json_path=(derivatives_root / row["review_json"]) if row.get("review_json") else None,
                reviewer=reviewer if reviewer else None,
                scan_id=None,
            )

    scope = f"scan_id={args.scan_id}" if args.scan_id else "all scans"
    who = f"reviewer={reviewer}" if reviewer else "all reviewers"
    print(f"[review-clear] cleared={total_cleared} ({scope}, {who})")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    output_path = Path(args.output).resolve() if args.output else get_export_path(derivatives_root)
    out = export_reviews(index_rows=index_rows, derivatives_root=derivatives_root, output_path=output_path)
    print(f"[export] wrote {out}")
    return 0


def _cmd_import_final_grades(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    stats = import_final_grades(
        index_rows=index_rows,
        derivatives_root=derivatives_root,
        import_path=Path(args.input).resolve(),
        reviewer=str(args.reviewer or "import").strip() or "import",
    )
    print(
        "[import-final-grades] "
        f"imported={stats['imported']} skipped_existing={stats['skipped_existing']} missing_scan={stats['missing_scan']}"
    )
    return 0


def _cmd_train_prepare(args: argparse.Namespace) -> int:
    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    output_path = Path(args.output).resolve() if args.output else (derivatives_root / "training" / "train_manifest.tsv")

    stats = build_training_manifest(
        derivatives_root=derivatives_root,
        output_path=output_path,
        min_auto_confidence=float(args.min_auto_confidence),
        slice_step=int(args.slice_step),
        slice_count=int(args.slice_count),
        include_auto_without_manual=bool(args.include_auto_without_manual),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        cv_folds=int(args.cv_folds),
        scaling=args.scaling,
    )
    print(
        "[train-prepare] "
        f"rows={stats['rows_written']} manual={stats['rows_manual']} auto={stats['rows_auto']} "
        f"split(train/val/test)={stats['split_train']}/{stats['split_val']}/{stats['split_test']} "
        f"cache(scans/slices)={stats.get('cache_scans', 0)}/{stats.get('cache_slices', 0)}"
    )
    print(f"[train-prepare] wrote manifest: {output_path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    output_model_dir = Path(args.output_model_dir).resolve()

    if args.init_model_dir:
        init_model_dir = Path(args.init_model_dir).resolve()
        init_model_id = Path(init_model_dir).name
    else:
        model_root = _resolve_model_root(args.model_root)
        model_id = str(args.init_model_id or "base-v1").strip()
        init_model_dir, profile = resolve_model_dir(model_root=model_root, model_id=model_id)
        init_model_id = str(profile.get("model_id", model_id)).strip()

    cfg = TrainConfig(
        manifest_path=manifest_path,
        init_model_dir=init_model_dir,
        output_model_dir=output_model_dir,
        device=args.device,
        scaling=args.scaling,
        batch_size=int(args.batch_size),
        epochs_head=int(args.epochs_head),
        epochs_finetune=int(args.epochs_finetune),
        lr_head=float(args.lr_head),
        lr_finetune=float(args.lr_finetune),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        aug_hflip=bool(args.aug_hflip),
        aug_vflip=bool(args.aug_vflip),
        aug_rotate=bool(args.aug_rotate),
        aug_crop=bool(args.aug_crop),
        max_cache_scans=int(args.max_cache_scans),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    summary = run_transfer_learning(cfg)
    print(
        "[train] "
        f"trained {len(summary.get('models', []))} model(s) from {init_model_id} "
        f"on {summary.get('n_rows', 0)} slices"
    )
    print(f"[train] output model dir: {output_model_dir}")
    print(f"[train] metrics: {output_model_dir / 'training_metrics.json'}")
    return 0


def _cmd_model_register(args: argparse.Namespace) -> int:
    model_root = _resolve_model_root(args.model_root)
    model_dir = Path(args.model_dir).resolve()
    registry_path, entry = register_model_profile(
        model_root=model_root,
        model_id=args.model_id,
        model_dir=model_dir,
        display_name=args.display_name,
        domain=args.domain,
        version=args.version,
        description=args.description,
        source_model_id=args.source_model_id,
        training_manifest=args.training_manifest,
        metrics_path=args.metrics_path,
        make_default=bool(args.make_default),
    )
    print(
        "[model-register] "
        f"model_id={entry.get('model_id')} checkpoints={entry.get('checkpoint_count')} "
        f"default={'yes' if bool(args.make_default) else 'no'}"
    )
    print(f"[model-register] registry: {registry_path}")
    return 0


def _cmd_model_list(args: argparse.Namespace) -> int:
    model_root = _resolve_model_root(args.model_root)
    payload = {
        "model_root": str(model_root),
        "registry_path": str(get_registry_path(model_root)),
        "models": list_model_profiles(model_root),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0

    models = payload.get("models", [])
    print(f"[model-list] model_root={model_root}")
    print(f"[model-list] discovered {len(models)} model profile(s)")
    for entry in models:
        mid = str(entry.get("model_id", "")).strip()
        version = str(entry.get("version", "")).strip()
        display = str(entry.get("display_name", "")).strip() or mid
        rel = str(entry.get("relative_dir", "")).strip()
        print(f"  - {display} ({mid}@{version}) -> {rel}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from motionscore.inference.model import ModelEnsemble
    try:
        from motionscore.explain.gradcam import generate_gradcam_attention_map
    except Exception as exc:
        raise RuntimeError(
            "Grad-CAM dependencies are optional and not installed. "
            "Install with: pip install -e '.[explain]'"
        ) from exc

    derivatives_root = _normalize_derivatives_root(Path(args.derivatives_root).resolve())
    index_rows = _read_index(derivatives_root)
    if not index_rows:
        raise FileNotFoundError(f"No index.tsv found in {derivatives_root}")

    hit = next((row for row in index_rows if row.get("scan_id") == args.scan_id), None)
    if hit is None:
        raise KeyError(f"scan_id not found in index.tsv: {args.scan_id}")

    existing_rel = hit.get("attention_map_path", "")
    if existing_rel and not args.overwrite:
        existing_path = derivatives_root / existing_rel
        if existing_path.exists():
            print(f"[explain] existing attention map: {existing_path}")
            return 0

    session = _session_from_index(hit)
    model_root = _resolve_model_root(args.model_root)
    model_id = str(getattr(args, "model_id", "") or hit.get("model_id", "") or "base-v1").strip()
    ensemble = ModelEnsemble(
        model_dir=args.model_dir,
        model_root=model_root,
        model_id=model_id,
        device=args.device,
    )
    print(f"[explain] using torch device={ensemble.model_device()}")
    aim_volume = read_aim(session.raw_image_path, scaling=args.scaling)
    prediction = predict_scan(
        aim_volume.data,
        ensemble=ensemble,
        stackheight=int(args.stackheight),
        on_incomplete_stack=args.on_incomplete_stack,
        slice_batch_size=int(args.slice_batch_size),
        retain_preprocessed=True,
    )

    explain_dir = get_explain_dir(derivatives_root, session)
    explain_dir.mkdir(parents=True, exist_ok=True)
    output_path = explain_dir / f"{args.scan_id}_gradcam.mha"

    _, written = generate_gradcam_attention_map(
        aim_volume=aim_volume,
        prediction=prediction,
        ensemble=ensemble,
        output_path=output_path,
    )
    if written is None:
        raise RuntimeError("failed to write attention map")

    hit["attention_map_path"] = to_relpath(written, derivatives_root)
    _upsert_index_rows(derivatives_root, [hit])

    print(f"[explain] wrote {written}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motionscore",
        description="MotionScore dataset pipeline (discover, predict, review, explain, export)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover AIM scans in flat/BIDS-style datasets")
    discover.add_argument("input_root", type=Path)
    discover.add_argument("--force-header-discovery", action="store_true")
    discover.add_argument("--json", dest="as_json", action="store_true")

    predict = subparsers.add_parser("predict", help="Run automatic prediction and initialize review files")
    predict.add_argument("input_root", type=Path)
    predict.add_argument("--output-root", type=Path, default=None)
    predict.add_argument("--model-dir", type=Path, default=None)
    predict.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Model registry root containing model_registry.json (defaults to ~/.motionscore/MotionScore/models).",
    )
    predict.add_argument(
        "--model-id",
        type=str,
        default="base-v1",
        help="Model profile id from model_registry.json (ignored when --model-dir is provided).",
    )
    predict.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    predict.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])
    predict.add_argument("--stackheight", type=int, default=168)
    predict.add_argument(
        "--slice-batch-size",
        type=int,
        default=64,
        help="How many preprocessed slices to run per inference batch (smaller uses less RAM).",
    )
    predict.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help="Process every n-th slice only (1=full scan, 2=every second slice, etc.).",
    )
    predict.add_argument("--on-incomplete-stack", type=str, default="keep_last", choices=["keep_last", "drop_last", "error"])
    predict.add_argument("--confidence-threshold", type=int, default=75)
    predict.add_argument(
        "--training-mode",
        action="store_true",
        help="Initialize review rows in blinded training mode (prediction hidden until manual grade is submitted).",
    )
    predict.add_argument(
        "--scan-id",
        action="append",
        default=None,
        help="Restrict prediction to one or more scan_id entries (repeat flag for multiple scans)",
    )
    predict.add_argument(
        "--manual-only",
        action="store_true",
        help="Initialize review pipeline without CNN inference (manual grading only).",
    )
    predict.add_argument("--preview-panels", type=int, default=3, help="Number of slice panels in preview PNG (1-9)")
    predict.add_argument("--preview-png", dest="save_preview_png", action="store_true", default=True, help="Save per-scan preview PNG into derivatives")
    predict.add_argument("--no-preview-png", dest="save_preview_png", action="store_false", help="Disable preview PNG export during predict")
    predict.add_argument("--force-header-discovery", action="store_true")

    review_init = subparsers.add_parser("review-init", help="Initialize or update review files from predictions")
    review_init.add_argument("derivatives_root", type=Path)
    review_init.add_argument("--confidence-threshold", type=int, default=75)
    review_init.add_argument(
        "--training-mode",
        action="store_true",
        help="Use blinded training mode for pending reviews.",
    )

    review_apply = subparsers.add_parser("review-apply", help="Apply a manual grade for one scan")
    review_apply.add_argument("derivatives_root", type=Path)
    review_apply.add_argument("--scan-id", required=True)
    review_apply.add_argument("--manual-grade", required=True, type=int, choices=[1, 2, 3, 4, 5])
    review_apply.add_argument("--reviewer", required=True)

    review_clear = subparsers.add_parser("review-clear", help="Clear manual grades for a reviewer (or all reviewers)")
    review_clear.add_argument("derivatives_root", type=Path)
    review_clear.add_argument("--scan-id", default=None, help="Optional: clear only one scan_id")
    clear_group = review_clear.add_mutually_exclusive_group(required=True)
    clear_group.add_argument("--reviewer", default=None, help="Reviewer id to clear")
    clear_group.add_argument("--all-reviewers", action="store_true", help="Clear grades regardless of reviewer")

    explain = subparsers.add_parser("explain", help="Generate on-demand Grad-CAM attention map for one scan")
    explain.add_argument("derivatives_root", type=Path)
    explain.add_argument("--scan-id", required=True)
    explain.add_argument("--model-dir", type=Path, default=None)
    explain.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Model registry root containing model_registry.json (defaults to ~/.motionscore/MotionScore/models).",
    )
    explain.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Model profile id from model_registry.json (defaults to scan model_id from index.tsv).",
    )
    explain.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    explain.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])
    explain.add_argument("--stackheight", type=int, default=168)
    explain.add_argument(
        "--slice-batch-size",
        type=int,
        default=64,
        help="How many preprocessed slices to run per inference batch while computing predictions.",
    )
    explain.add_argument("--on-incomplete-stack", type=str, default="keep_last", choices=["keep_last", "drop_last", "error"])
    explain.add_argument("--overwrite", action="store_true")

    export = subparsers.add_parser("export", help="Export final grades across all scans")
    export.add_argument("derivatives_root", type=Path)
    export.add_argument("--output", type=Path, default=None)

    import_final = subparsers.add_parser("import-final-grades", help="Import final grades and fill only missing manual/final grades")
    import_final.add_argument("derivatives_root", type=Path)
    import_final.add_argument("--input", required=True, type=Path)
    import_final.add_argument("--reviewer", default="import")

    train_prepare = subparsers.add_parser(
        "train-prepare",
        help="Build slice-level training manifest from review/prediction outputs (manual labels take priority).",
    )
    train_prepare.add_argument("derivatives_root", type=Path)
    train_prepare.add_argument("--output", type=Path, default=None)
    train_prepare.add_argument("--min-auto-confidence", type=float, default=0.70)
    train_prepare.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help="Use every n-th slice when creating the training manifest and slice cache DB.",
    )
    train_prepare.add_argument(
        "--slice-count",
        type=int,
        default=8,
        help="Randomized per-scan slice sample count (0 disables and uses --slice-step).",
    )
    train_prepare.add_argument("--include-auto-without-manual", action="store_true", default=False)
    train_prepare.add_argument("--seed", type=int, default=13)
    train_prepare.add_argument("--cv-folds", type=int, default=10)
    train_prepare.add_argument("--train-ratio", type=float, default=0.70)
    train_prepare.add_argument("--val-ratio", type=float, default=0.15)
    train_prepare.add_argument("--test-ratio", type=float, default=0.15)
    train_prepare.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])

    train = subparsers.add_parser(
        "train",
        help="Run PyTorch transfer learning from base checkpoints using a prepared slice manifest.",
    )
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output-model-dir", required=True, type=Path)
    train.add_argument("--init-model-dir", type=Path, default=None)
    train.add_argument("--model-root", type=Path, default=None)
    train.add_argument("--init-model-id", type=str, default="base-v1")
    train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    train.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])
    train.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--epochs-head", type=int, default=20)
    train.add_argument("--epochs-finetune", type=int, default=50)
    train.add_argument("--lr-head", type=float, default=1e-3)
    train.add_argument("--lr-finetune", type=float, default=1e-4)
    train.add_argument("--early-stopping-patience", type=int, default=10)
    train.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    train.add_argument("--aug-hflip", dest="aug_hflip", action="store_true", default=True)
    train.add_argument("--no-aug-hflip", dest="aug_hflip", action="store_false")
    train.add_argument("--aug-vflip", dest="aug_vflip", action="store_true", default=True)
    train.add_argument("--no-aug-vflip", dest="aug_vflip", action="store_false")
    train.add_argument("--aug-rotate", dest="aug_rotate", action="store_true", default=False)
    train.add_argument("--no-aug-rotate", dest="aug_rotate", action="store_false")
    train.add_argument("--aug-crop", dest="aug_crop", action="store_true", default=False)
    train.add_argument("--no-aug-crop", dest="aug_crop", action="store_false")
    train.add_argument("--max-cache-scans", type=int, default=64)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=13)

    model_register = subparsers.add_parser("model-register", help="Register a model profile in model_registry.json")
    model_register.add_argument("--model-root", type=Path, default=None)
    model_register.add_argument("--model-id", required=True)
    model_register.add_argument("--model-dir", required=True, type=Path)
    model_register.add_argument("--display-name", required=True)
    model_register.add_argument("--domain", type=str, default="custom")
    model_register.add_argument("--version", type=str, default="v1")
    model_register.add_argument("--description", type=str, default="")
    model_register.add_argument("--source-model-id", type=str, default="")
    model_register.add_argument("--training-manifest", type=str, default="")
    model_register.add_argument("--metrics-path", type=str, default="")
    model_register.add_argument("--make-default", action="store_true")

    model_list = subparsers.add_parser("model-list", help="List registered model profiles from model_registry.json")
    model_list.add_argument("--model-root", type=Path, default=None)
    model_list.add_argument("--json", dest="as_json", action="store_true")

    return parser


def main() -> None:
    _print_citation_notice()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "discover":
            raise SystemExit(_cmd_discover(args))
        if args.command == "predict":
            raise SystemExit(_cmd_predict(args))
        if args.command == "review-init":
            raise SystemExit(_cmd_review_init(args))
        if args.command == "review-apply":
            raise SystemExit(_cmd_review_apply(args))
        if args.command == "review-clear":
            raise SystemExit(_cmd_review_clear(args))
        if args.command == "explain":
            raise SystemExit(_cmd_explain(args))
        if args.command == "export":
            raise SystemExit(_cmd_export(args))
        if args.command == "import-final-grades":
            raise SystemExit(_cmd_import_final_grades(args))
        if args.command == "train-prepare":
            raise SystemExit(_cmd_train_prepare(args))
        if args.command == "train":
            raise SystemExit(_cmd_train(args))
        if args.command == "model-register":
            raise SystemExit(_cmd_model_register(args))
        if args.command == "model-list":
            raise SystemExit(_cmd_model_list(args))

        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[motionscore] error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
