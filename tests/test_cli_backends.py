from motionscore.cli import _build_parser


def test_predict_backend_choices() -> None:
    parser = _build_parser()
    args = parser.parse_args(["predict", "/tmp/data", "--backend", "torch"])
    assert args.command == "predict"
    assert args.backend == "torch"


def test_explain_backend_choices() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "explain",
            "/tmp/data/derivatives/MotionScore",
            "--scan-id",
            "scan_1",
            "--backend",
            "tensorflow",
        ]
    )
    assert args.command == "explain"
    assert args.backend == "tensorflow"


def test_convert_torch_command_exists() -> None:
    parser = _build_parser()
    args = parser.parse_args(["convert-torch"])
    assert args.command == "convert-torch"


def test_predict_preview_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["predict", "/tmp/data", "--preview-panels", "5", "--no-preview-png"]
    )
    assert args.command == "predict"
    assert args.preview_panels == 5
    assert args.save_preview_png is False
    assert args.slice_batch_size == 64


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
