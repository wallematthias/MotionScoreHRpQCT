from motionscore.cli import _build_parser


def test_predict_parses_without_backend_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["predict", "/tmp/data"])
    assert args.command == "predict"
    assert args.model_dir is None
    assert args.model_id == "base-v1"


def test_explain_parses_without_backend_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "explain",
            "/tmp/data/derivatives/MotionScore",
            "--scan-id",
            "scan_1",
        ]
    )
    assert args.command == "explain"
    assert args.scan_id == "scan_1"
    assert args.model_id == ""


def test_predict_preview_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["predict", "/tmp/data", "--preview-panels", "5", "--no-preview-png"]
    )
    assert args.command == "predict"
    assert args.preview_panels == 5
    assert args.save_preview_png is False
    assert args.slice_batch_size == 64
    assert args.slice_step == 1


def test_predict_slice_step_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["predict", "/tmp/data", "--slice-step", "3"])
    assert args.command == "predict"
    assert args.slice_step == 3


def test_predict_manual_only_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["predict", "/tmp/data", "--manual-only"])
    assert args.command == "predict"
    assert args.manual_only is True


def test_predict_scan_id_filter_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "predict",
            "/tmp/data",
            "--scan-id",
            "scan_a",
            "--scan-id",
            "scan_b",
        ]
    )
    assert args.command == "predict"
    assert args.scan_id == ["scan_a", "scan_b"]


def test_training_mode_flags() -> None:
    parser = _build_parser()
    args_predict = parser.parse_args(
        [
            "predict",
            "/tmp/data",
            "--training-mode",
        ]
    )
    assert args_predict.command == "predict"
    assert args_predict.training_mode is True

    args_review = parser.parse_args(
        [
            "review-init",
            "/tmp/data/derivatives/MotionScore",
            "--training-mode",
        ]
    )
    assert args_review.command == "review-init"
    assert args_review.training_mode is True


def test_review_clear_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "review-clear",
            "/tmp/data/derivatives/MotionScore",
            "--reviewer",
            "opA",
        ]
    )
    assert args.command == "review-clear"
    assert args.reviewer == "opA"
    assert args.all_reviewers is False

    args_all = parser.parse_args(
        [
            "review-clear",
            "/tmp/data/derivatives/MotionScore",
            "--all-reviewers",
        ]
    )
    assert args_all.command == "review-clear"
    assert args_all.all_reviewers is True


def test_import_final_grades_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "import-final-grades",
            "/tmp/data/derivatives/MotionScore",
            "--input",
            "/tmp/final.tsv",
            "--reviewer",
            "importer",
        ]
    )
    assert args.command == "import-final-grades"
    assert str(args.input) == "/tmp/final.tsv"
    assert args.reviewer == "importer"


def test_train_prepare_and_model_commands_parse() -> None:
    parser = _build_parser()
    args_prepare = parser.parse_args(["train-prepare", "/tmp/data/derivatives/MotionScore"])
    assert args_prepare.command == "train-prepare"
    assert args_prepare.min_auto_confidence == 0.70
    assert args_prepare.slice_step == 1
    assert args_prepare.slice_count == 8
    assert args_prepare.include_auto_without_manual is False

    args_train = parser.parse_args(
        [
            "train",
            "--manifest",
            "/tmp/manifest.tsv",
            "--output-model-dir",
            "/tmp/new_model",
        ]
    )
    assert args_train.command == "train"
    assert args_train.init_model_id == "base-v1"
    assert args_train.early_stopping_patience == 10
    assert args_train.aug_hflip is True
    assert args_train.aug_vflip is True
    assert args_train.aug_rotate is False
    assert args_train.aug_crop is False

    args_register = parser.parse_args(
        [
            "model-register",
            "--model-id",
            "knee-v1",
            "--model-dir",
            "/tmp/new_model",
            "--display-name",
            "Knee v1",
        ]
    )
    assert args_register.command == "model-register"

    args_list = parser.parse_args(["model-list", "--json"])
    assert args_list.command == "model-list"
    assert args_list.as_json is True
