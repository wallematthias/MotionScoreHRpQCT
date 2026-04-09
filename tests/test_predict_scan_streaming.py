import numpy as np
import pytest

from motionscore.inference.scoring import predict_scan


class DummyEnsemble:
    def __init__(self, n_models: int = 2):
        self.n_models = n_models
        self.calls: list[int] = []

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(int(batch.shape[0]))
        n = int(batch.shape[0])
        out = np.zeros((self.n_models, n, 5), dtype=np.float32)
        out[:, :, 1] = 0.95
        out[:, :, 0] = 0.05
        return out


def test_predict_scan_streams_slice_batches() -> None:
    volume = np.random.RandomState(0).rand(16, 16, 10).astype(np.float32)
    ensemble = DummyEnsemble()

    result = predict_scan(
        volume,
        ensemble=ensemble,
        stackheight=4,
        slice_batch_size=3,
        retain_preprocessed=False,
    )

    assert ensemble.calls == [3, 3, 3, 1]
    assert result.preprocessed_scan is None
    assert result.preprocess_infos == []
    assert len(result.slice_grades) == 10
    assert len(result.slice_confidences) == 10
    assert result.automatic_grade == 2


def test_predict_scan_can_retain_preprocessed_for_explain() -> None:
    volume = np.random.RandomState(1).rand(12, 12, 7).astype(np.float32)
    ensemble = DummyEnsemble()

    result = predict_scan(
        volume,
        ensemble=ensemble,
        stackheight=4,
        slice_batch_size=2,
        retain_preprocessed=True,
    )

    assert ensemble.calls == [2, 2, 2, 1]
    assert result.preprocessed_scan is not None
    assert result.preprocessed_scan.shape[0] == 7
    assert len(result.preprocess_infos) == 7


def test_predict_scan_slice_step_sparsifies_outputs() -> None:
    volume = np.random.RandomState(2).rand(16, 16, 9).astype(np.float32)
    ensemble = DummyEnsemble()

    result = predict_scan(
        volume,
        ensemble=ensemble,
        stackheight=4,
        slice_batch_size=2,
        slice_step=3,
        retain_preprocessed=False,
    )

    assert ensemble.calls == [2, 1]
    assert len(result.slice_grades) == 9
    assert len(result.slice_confidences) == 9
    assert result.slice_grades == [2, 0, 0, 2, 0, 0, 2, 0, 0]
    assert result.slice_confidences[0] > 0
    assert result.slice_confidences[1] < 0
    assert result.slice_confidences[3] > 0
    assert result.votes.shape == (9, 5)
    assert np.allclose(result.votes[1], np.zeros(5, dtype=np.float32))


def test_predict_scan_retain_preprocessed_requires_full_slice_step() -> None:
    volume = np.random.RandomState(3).rand(8, 8, 5).astype(np.float32)
    ensemble = DummyEnsemble()

    with pytest.raises(ValueError, match="slice_step must be 1"):
        predict_scan(
            volume,
            ensemble=ensemble,
            stackheight=4,
            slice_batch_size=2,
            slice_step=2,
            retain_preprocessed=True,
        )
