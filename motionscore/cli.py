from __future__ import annotations

import argparse
import gc
import json
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
from motionscore.review.store import (
    apply_manual_review,
    clear_manual_reviews,
    export_reviews,
    initialize_or_update_review,
)
from motionscore.review.preview import write_prediction_preview_png, write_slice_profile_png
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
    "site",
    "session_id",
    "stack_index",
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
    "model_version",
    "predicted_at",
]

PREDICTION_FIELDS = [
    "scan_id",
    "subject_id",
    "site",
    "session_id",
    "stack_index",
    "raw_image_path",
    "preview_png_path",
    "slice_profile_png_path",
    "automatic_grade",
    "automatic_confidence",
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
        subject_id=row["subject_id"],
        site=row["site"],
        session_id=row["session_id"],
        raw_image_path=Path(row["raw_image_path"]),
        stack_index=int(row["stack_index"]) if row.get("stack_index") else None,
    )


def _scan_id_for_session(session: RawSession) -> str:
    return make_scan_id(session.subject_id, session.site, session.session_id, session.raw_image_path)


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

    ensemble = ModelEnsemble(args.model_dir, device=args.device)
    model_version = ensemble.model_version()
    print(f"[predict] using torch device={ensemble.model_device()}")

    index_updates: list[dict[str, str]] = []
    for session in sessions:
        scan_id = _scan_id_for_session(session)
        aim_volume = read_aim(session.raw_image_path, scaling=args.scaling)
        prediction = predict_scan(
            aim_volume.data,
            ensemble=ensemble,
            stackheight=int(args.stackheight),
            on_incomplete_stack=args.on_incomplete_stack,
            slice_batch_size=int(args.slice_batch_size),
            retain_preprocessed=False,
        )

        predicted_at = utc_now_iso()

        pred_dir = get_predictions_dir(derivatives_root, session)
        review_dir = get_review_dir(derivatives_root, session)

        predictions_tsv = pred_dir / "predictions.tsv"
        review_tsv = review_dir / "review.tsv"
        review_json = review_dir / "review.json"
        review_audit = review_dir / "review_audit.tsv"

        prediction_row = {
            "scan_id": scan_id,
            "subject_id": session.subject_id,
            "site": session.site,
            "session_id": session.session_id,
            "stack_index": "" if session.stack_index is None else str(session.stack_index),
            "raw_image_path": str(session.raw_image_path.resolve()),
            "preview_png_path": "",
            "slice_profile_png_path": "",
            "automatic_grade": str(prediction.automatic_grade),
            "automatic_confidence": str(prediction.automatic_confidence),
            "mean_confidence": str(prediction.mean_confidence),
            "stack_ranges": json.dumps(prediction.stack_ranges),
            "stack_grades": json.dumps(prediction.stack_grades),
            "stack_confidences": json.dumps(prediction.stack_confidences),
            "slice_grades": json.dumps(prediction.slice_grades),
            "slice_confidences": json.dumps(prediction.slice_confidences),
            "model_version": model_version,
            "predicted_at": predicted_at,
        }

        if bool(args.save_preview_png):
            preview_dir = get_preview_dir(derivatives_root, session)
            preview_path = preview_dir / f"{scan_id}_preview.png"
            profile_path = preview_dir / f"{scan_id}_slice_profile.png"
            written_preview = write_prediction_preview_png(
                volume_xyz=aim_volume.data,
                prediction=prediction,
                output_path=preview_path,
                max_panels=preview_panels,
            )
            prediction_row["preview_png_path"] = to_relpath(written_preview, derivatives_root)
            try:
                written_profile = write_slice_profile_png(
                    prediction=prediction,
                    output_path=profile_path,
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
                "site": session.site,
                "session_id": session.session_id,
                "stack_index": "" if session.stack_index is None else str(session.stack_index),
                "raw_image_path": str(session.raw_image_path.resolve()),
                "predictions_tsv": to_relpath(predictions_tsv, derivatives_root),
                "review_tsv": to_relpath(review_tsv, derivatives_root),
                "review_json": to_relpath(review_json, derivatives_root),
                "review_audit": to_relpath(review_audit, derivatives_root),
                "preview_png_path": prediction_row.get("preview_png_path", ""),
                "slice_profile_png_path": prediction_row.get("slice_profile_png_path", ""),
                "attention_map_path": "",
                "automatic_grade": str(prediction.automatic_grade),
                "automatic_confidence": str(prediction.automatic_confidence),
                "model_version": model_version,
                "predicted_at": predicted_at,
            }
        )

        print(
            f"[predict] {scan_id} grade={prediction.automatic_grade} conf={prediction.automatic_confidence}%"
        )

        # Keep memory bounded across large datasets by releasing per-scan arrays early.
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
            "site": row.get("site", ""),
            "session_id": row.get("session_id", ""),
            "automatic_grade": row.get("automatic_grade", ""),
            "automatic_confidence": row.get("automatic_confidence", ""),
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
    ensemble = ModelEnsemble(args.model_dir, device=args.device)
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
    predict.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    predict.add_argument("--scaling", type=str, default="native", choices=["native", "none", "mu", "hu", "bmd", "density"])
    predict.add_argument("--stackheight", type=int, default=168)
    predict.add_argument(
        "--slice-batch-size",
        type=int,
        default=64,
        help="How many preprocessed slices to run per inference batch (smaller uses less RAM).",
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

        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[motionscore] error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
