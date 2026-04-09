from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import motionscore.cli as cli
from motionscore.dataset.models import RawSession
from motionscore.utils import read_tsv


class _FakeEnsemble:
    def __init__(self, model_dir=None, model_root=None, model_id=None, device="auto"):
        self.model_dir = model_dir
        self.model_root = model_root
        self.model_id = model_id or "base-v1"
        self.device = device

    def model_version(self):
        return "DNN_0.pt+DNN_1.pt"

    def model_identity(self):
        return f"{self.model_id}:DNN_0.pt+DNN_1.pt"

    def resolved_model_id(self):
        return self.model_id

    def model_device(self):
        return self.device


class _FakePrediction:
    automatic_grade = 3
    automatic_confidence = 82
    mean_confidence = 0.82
    stack_ranges = [(0, 4)]
    stack_grades = [3.0]
    stack_confidences = [0.82]
    slice_grades = [3, 3, 3, 3]
    slice_confidences = [0.8, 0.8, 0.85, 0.83]
    preprocessed_scan = None


def _predict_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_root=root,
        output_root=None,
        model_dir=None,
        model_root=None,
        model_id="base-v1",
        device="cpu",
        scaling="native",
        stackheight=4,
        slice_batch_size=4,
        slice_step=1,
        on_incomplete_stack="keep_last",
        confidence_threshold=75,
        training_mode=False,
        manual_only=False,
        scan_id=None,
        preview_panels=3,
        save_preview_png=True,
        force_header_discovery=False,
    )


def test_normalize_derivatives_root(tmp_path: Path) -> None:
    plain = tmp_path / "x"
    assert cli._normalize_derivatives_root(plain) == plain

    d = tmp_path / "derivatives" / "MotionScore"
    d.mkdir(parents=True)
    assert cli._normalize_derivatives_root(tmp_path) == d
    assert cli._normalize_derivatives_root(d) == d


def test_cmd_discover_text_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    sessions = [
        RawSession("SUB1", "tibia", "T1", tmp_path / "SUB1_DT_T1.AIM"),
        RawSession("SUB2", "radius", "T2", tmp_path / "SUB2_DR_T2.AIM", stack_index=2),
    ]
    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: sessions)

    args_json = argparse.Namespace(input_root=tmp_path, force_header_discovery=False, as_json=True)
    assert cli._cmd_discover(args_json) == 0
    out_json = capsys.readouterr().out
    assert '"subject_id": "SUB1"' in out_json

    args_txt = argparse.Namespace(input_root=tmp_path, force_header_discovery=False, as_json=False)
    assert cli._cmd_discover(args_txt) == 0
    out_txt = capsys.readouterr().out
    assert "[motionscore] discovered 2 scan(s)" in out_txt


def test_cmd_predict_and_review_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    aim_path = root / "SUB1_DT_T1.AIM"
    aim_path.write_bytes(b"aim")
    session = RawSession("SUB1", "tibia", "T1", aim_path)

    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: [session])
    monkeypatch.setattr(cli, "read_aim", lambda _p, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)))
    monkeypatch.setattr(cli, "predict_scan", lambda *_args, **_kwargs: _FakePrediction())
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        _FakeEnsemble,
    )

    def _write_preview(volume_xyz, prediction, output_path, max_panels=3):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path

    def _write_profile(prediction, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"profile")
        return output_path

    monkeypatch.setattr(cli, "write_prediction_preview_png", _write_preview)
    monkeypatch.setattr(cli, "write_slice_profile_png", _write_profile)

    args = _predict_args(root)
    assert cli._cmd_predict(args) == 0

    derivatives = root / "derivatives" / "MotionScore"
    index_rows = read_tsv(derivatives / "index.tsv")
    assert len(index_rows) == 1
    scan_id = index_rows[0]["scan_id"]
    assert "site" not in index_rows[0]
    assert "session_id" not in index_rows[0]
    assert "stack_index" not in index_rows[0]

    args_init = argparse.Namespace(derivatives_root=derivatives, confidence_threshold=70, training_mode=True)
    assert cli._cmd_review_init(args_init) == 0

    args_apply = argparse.Namespace(
        derivatives_root=derivatives,
        scan_id=scan_id,
        manual_grade=4,
        reviewer="op_a",
    )
    assert cli._cmd_review_apply(args_apply) == 0

    args_clear = argparse.Namespace(
        derivatives_root=derivatives,
        scan_id=None,
        reviewer="op_a",
        all_reviewers=False,
    )
    assert cli._cmd_review_clear(args_clear) == 0

    args_export = argparse.Namespace(derivatives_root=derivatives, output=None)
    assert cli._cmd_export(args_export) == 0
    assert (derivatives / "motion_grades.tsv").exists()


def test_cmd_predict_profile_png_missing_dependency(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    aim_path = root / "SUB1_DT_T1.AIM"
    aim_path.write_bytes(b"aim")
    session = RawSession("SUB1", "tibia", "T1", aim_path)

    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: [session])
    monkeypatch.setattr(cli, "read_aim", lambda _p, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)))
    monkeypatch.setattr(cli, "predict_scan", lambda *_args, **_kwargs: _FakePrediction())
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        _FakeEnsemble,
    )

    monkeypatch.setattr(
        cli,
        "write_prediction_preview_png",
        lambda volume_xyz, prediction, output_path, max_panels=3: output_path,
    )
    monkeypatch.setattr(
        cli,
        "write_slice_profile_png",
        lambda prediction, output_path: (_ for _ in ()).throw(RuntimeError("matplotlib missing")),
    )

    args = _predict_args(root)
    assert cli._cmd_predict(args) == 0
    err = capsys.readouterr().err
    assert "skipping slice profile PNG" in err


def test_cmd_predict_manual_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    aim_path = root / "randomfilename.AIM"
    aim_path.write_bytes(b"aim")
    session = RawSession("randomfilename", "tibia", "T1", aim_path)

    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: [session])
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ModelEnsemble should not be used in manual-only mode")),
    )

    args = _predict_args(root)
    args.manual_only = True
    assert cli._cmd_predict(args) == 0

    derivatives = root / "derivatives" / "MotionScore"
    index_rows = read_tsv(derivatives / "index.tsv")
    assert len(index_rows) == 1
    assert index_rows[0]["manual_mode"] == "1"
    assert index_rows[0]["automatic_grade"] == ""

    pred_tsv = derivatives / index_rows[0]["predictions_tsv"]
    pred_rows = read_tsv(pred_tsv)
    assert len(pred_rows) == 1
    assert pred_rows[0]["manual_mode"] == "1"
    assert pred_rows[0]["automatic_confidence"] == ""


def test_cmd_predict_manual_only_preserves_existing_ai_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    aim_path = root / "randomfilename.AIM"
    aim_path.write_bytes(b"aim")
    session = RawSession("randomfilename", "tibia", "T1", aim_path)

    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: [session])
    monkeypatch.setattr(cli, "read_aim", lambda _p, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)))
    monkeypatch.setattr(cli, "predict_scan", lambda *_args, **_kwargs: _FakePrediction())
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        _FakeEnsemble,
    )
    monkeypatch.setattr(
        cli,
        "write_prediction_preview_png",
        lambda volume_xyz, prediction, output_path, max_panels=3: output_path,
    )
    monkeypatch.setattr(
        cli,
        "write_slice_profile_png",
        lambda prediction, output_path: output_path,
    )

    args = _predict_args(root)
    args.save_preview_png = False
    assert cli._cmd_predict(args) == 0

    # Manual-only run should keep existing AI fields in predictions.tsv if present.
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ModelEnsemble should not be used in manual-only mode")),
    )
    args2 = _predict_args(root)
    args2.manual_only = True
    args2.save_preview_png = False
    assert cli._cmd_predict(args2) == 0

    derivatives = root / "derivatives" / "MotionScore"
    index_rows = read_tsv(derivatives / "index.tsv")
    assert len(index_rows) == 1
    assert index_rows[0]["manual_mode"] == "1"
    assert index_rows[0]["automatic_grade"] == "3"

    pred_tsv = derivatives / index_rows[0]["predictions_tsv"]
    pred_rows = read_tsv(pred_tsv)
    assert len(pred_rows) == 1
    assert pred_rows[0]["manual_mode"] == "1"
    assert pred_rows[0]["automatic_grade"] == "3"
    assert pred_rows[0]["slice_grades"] != ""


def test_cmd_predict_stores_outputs_side_by_side_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    aim_path = root / "SUB1_DT_T1.AIM"
    aim_path.write_bytes(b"aim")
    session = RawSession("SUB1", "tibia", "T1", aim_path)

    monkeypatch.setattr(cli, "discover_raw_sessions", lambda **_kwargs: [session])
    monkeypatch.setattr(cli, "read_aim", lambda _p, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)))
    monkeypatch.setattr(cli, "predict_scan", lambda *_args, **_kwargs: _FakePrediction())
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        _FakeEnsemble,
    )
    monkeypatch.setattr(
        cli,
        "write_prediction_preview_png",
        lambda volume_xyz, prediction, output_path, max_panels=3: output_path,
    )
    monkeypatch.setattr(
        cli,
        "write_slice_profile_png",
        lambda prediction, output_path: output_path,
    )

    args_a = _predict_args(root)
    args_a.model_id = "base-v1"
    args_a.save_preview_png = False
    assert cli._cmd_predict(args_a) == 0

    args_b = _predict_args(root)
    args_b.model_id = "knee-v1"
    args_b.save_preview_png = False
    assert cli._cmd_predict(args_b) == 0

    derivatives = root / "derivatives" / "MotionScore"
    base_pred = derivatives / "sub-SUB1" / "site-tibia" / "ses-T1" / "predictions" / "models" / "base-v1" / "predictions.tsv"
    knee_pred = derivatives / "sub-SUB1" / "site-tibia" / "ses-T1" / "predictions" / "models" / "knee-v1" / "predictions.tsv"
    assert base_pred.exists()
    assert knee_pred.exists()

    base_rows = read_tsv(base_pred)
    knee_rows = read_tsv(knee_pred)
    assert base_rows[0]["model_id"] == "base-v1"
    assert knee_rows[0]["model_id"] == "knee-v1"

    index_rows = read_tsv(derivatives / "index.tsv")
    assert len(index_rows) == 1
    assert "predictions/models/knee-v1/predictions.tsv" in index_rows[0]["predictions_tsv"]


def test_cmd_explain_existing_and_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    derivatives = tmp_path / "derivatives" / "MotionScore"
    derivatives.mkdir(parents=True)
    (derivatives / "dataset_description.json").write_text("{}", encoding="utf-8")

    review_dir = derivatives / "sub-SUB1" / "site-tibia" / "ses-T1" / "review"
    review_dir.mkdir(parents=True)
    (review_dir / "review.tsv").write_text("scan_id\nx\n", encoding="utf-8")
    (review_dir / "review.json").write_text("{}", encoding="utf-8")
    (review_dir / "review_audit.tsv").write_text("scan_id\tevent\nx\tinit\n", encoding="utf-8")

    existing_map = derivatives / "sub-SUB1" / "site-tibia" / "ses-T1" / "explain" / "x_gradcam.mha"
    existing_map.parent.mkdir(parents=True)
    existing_map.write_bytes(b"mha")

    index = derivatives / "index.tsv"
    index.write_text(
        "\t".join(cli.INDEX_FIELDS)
        + "\n"
        + "\t".join(
            [
                "x",
                "SUB1",
                str((tmp_path / "raw.AIM").resolve()),
                "sub-SUB1/site-tibia/ses-T1/predictions/predictions.tsv",
                "sub-SUB1/site-tibia/ses-T1/review/review.tsv",
                "sub-SUB1/site-tibia/ses-T1/review/review.json",
                "sub-SUB1/site-tibia/ses-T1/review/review_audit.tsv",
                "",
                "",
                str(existing_map.relative_to(derivatives)),
                "3",
                "82",
                "0",
                "base-v1",
                "DNN_0.pt",
                "2026-01-01T00:00:00+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Existing map path branch
    args_existing = argparse.Namespace(
        derivatives_root=derivatives,
        scan_id="x",
        model_dir=None,
        model_root=None,
        model_id="",
        device="cpu",
        scaling="native",
        stackheight=4,
        slice_batch_size=4,
        on_incomplete_stack="keep_last",
        overwrite=False,
    )
    assert cli._cmd_explain(args_existing) == 0

    # Overwrite branch with mocked inference + gradcam
    monkeypatch.setattr(
        __import__("motionscore.inference.model", fromlist=["ModelEnsemble"]),
        "ModelEnsemble",
        _FakeEnsemble,
    )
    monkeypatch.setattr(cli, "read_aim", lambda _p, scaling="native": SimpleNamespace(data=np.zeros((8, 8, 4), dtype=np.float32)))
    pred = _FakePrediction()
    pred.preprocessed_scan = np.zeros((4, 512, 512, 1), dtype=np.float32)
    pred.preprocess_infos = [SimpleNamespace(orig_h=8, orig_w=8, square_size=8, top_pad=0, bottom_pad=0, left_pad=0, right_pad=0)] * 4
    monkeypatch.setattr(cli, "predict_scan", lambda *_args, **_kwargs: pred)
    monkeypatch.setattr(
        __import__("motionscore.explain.gradcam", fromlist=["generate_gradcam_attention_map"]),
        "generate_gradcam_attention_map",
        lambda aim_volume, prediction, ensemble, output_path: (np.zeros((8, 8, 4), dtype=np.float32), output_path),
    )

    args_overwrite = argparse.Namespace(
        derivatives_root=derivatives,
        scan_id="x",
        model_dir=None,
        model_root=None,
        model_id="",
        device="cpu",
        scaling="native",
        stackheight=4,
        slice_batch_size=4,
        on_incomplete_stack="keep_last",
        overwrite=True,
    )
    assert cli._cmd_explain(args_overwrite) == 0


def test_main_wraps_runtime_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(command="predict")

    monkeypatch.setattr(cli, "_build_parser", lambda: _FakeParser())
    monkeypatch.setattr(cli, "_print_citation_notice", lambda: None)
    monkeypatch.setattr(cli, "_cmd_predict", lambda _args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "[motionscore] error: boom" in capsys.readouterr().err
