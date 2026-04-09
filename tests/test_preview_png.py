from pathlib import Path

import numpy as np
from PIL import Image

from motionscore.inference.scoring import PredictionResult
from motionscore.review.preview import write_prediction_preview_png, write_slice_profile_png


def test_write_prediction_preview_png(tmp_path: Path) -> None:
    volume = np.zeros((32, 28, 7), dtype=np.float32)
    for z in range(volume.shape[2]):
        volume[:, :, z] = float(z) * 10.0

    prediction = PredictionResult(
        automatic_grade=3,
        automatic_confidence=88,
        mean_confidence=0.88,
        slice_grades=[3] * volume.shape[2],
        slice_confidences=[0.88] * volume.shape[2],
        stack_grades=[3.0],
        stack_confidences=[0.88],
        stack_ranges=[(0, volume.shape[2])],
        preprocessed_scan=np.zeros((volume.shape[2], 512, 512, 1), dtype=np.float32),
        preprocess_infos=[],
        votes=np.zeros((volume.shape[2], 5), dtype=np.float32),
    )

    output = tmp_path / "preview.png"
    written = write_prediction_preview_png(volume_xyz=volume, prediction=prediction, output_path=output, max_panels=3)

    assert written == output
    assert written.exists()
    with Image.open(written) as img:
        assert img.format == "PNG"
        assert img.size[0] > 256
        assert img.size[1] > 256


def test_write_slice_profile_png(tmp_path: Path) -> None:
    prediction = PredictionResult(
        automatic_grade=2,
        automatic_confidence=76,
        mean_confidence=0.76,
        slice_grades=[1, 2, 2, 3, 2, 2, 1],
        slice_confidences=[0.62, 0.71, 0.84, 0.78, 0.81, 0.73, 0.59],
        stack_grades=[2.0],
        stack_confidences=[0.76],
        stack_ranges=[(0, 7)],
        preprocessed_scan=np.zeros((7, 512, 512, 1), dtype=np.float32),
        preprocess_infos=[],
        votes=np.zeros((7, 5), dtype=np.float32),
    )

    output = tmp_path / "slice_profile.png"
    written = write_slice_profile_png(prediction=prediction, output_path=output)
    assert written == output
    assert written.exists()
    with Image.open(written) as img:
        assert img.format == "PNG"
        assert img.size[0] > 512
        assert img.size[1] > 150


def test_write_slice_profile_png_with_sparse_predictions(tmp_path: Path) -> None:
    prediction = PredictionResult(
        automatic_grade=2,
        automatic_confidence=76,
        mean_confidence=0.76,
        slice_grades=[2, 0, 2, 0, 2],
        slice_confidences=[0.8, -1.0, 0.75, -1.0, 0.7],
        stack_grades=[2.0],
        stack_confidences=[0.76],
        stack_ranges=[(0, 5)],
        preprocessed_scan=None,
        preprocess_infos=[],
        votes=np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )

    output = tmp_path / "slice_profile_sparse.png"
    written = write_slice_profile_png(prediction=prediction, output_path=output)
    assert written.exists()
