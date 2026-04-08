import numpy as np

from motionscore.explain import gradcam as gc
from motionscore.inference.preprocessing import PreprocessInfo
from motionscore.inference.scoring import PredictionResult
from motionscore.io.aim import AimVolume


class _FakeEnsemble:
    def grad_models(self):
        return [object(), object()]


def test_generate_gradcam_attention_map_with_mocked_slice_gradcam(monkeypatch) -> None:
    def fake_slice_gradcam(_grad_model, _image_tensor, _class_idx):
        return np.ones((512, 512), dtype=np.float32)

    monkeypatch.setattr(gc, "_slice_gradcam_torch", fake_slice_gradcam)

    volume = np.zeros((8, 6, 3), dtype=np.float32)
    infos = [
        PreprocessInfo(orig_h=8, orig_w=6, square_size=8, top_pad=0, bottom_pad=0, left_pad=1, right_pad=1)
        for _ in range(3)
    ]
    prediction = PredictionResult(
        automatic_grade=3,
        automatic_confidence=90,
        mean_confidence=0.9,
        slice_grades=[3, 3, 3],
        slice_confidences=[0.9, 0.9, 0.9],
        stack_grades=[3.0],
        stack_confidences=[0.9],
        stack_ranges=[(0, 3)],
        preprocessed_scan=np.zeros((3, 512, 512, 1), dtype=np.float32),
        preprocess_infos=infos,
        votes=np.zeros((3, 5), dtype=np.float32),
    )
    aim = AimVolume(
        data=volume,
        spacing=(0.1, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        processing_log="",
        unit="native",
    )

    attention, path = gc.generate_gradcam_attention_map(
        aim_volume=aim,
        prediction=prediction,
        ensemble=_FakeEnsemble(),
        output_path=None,
    )
    assert path is None
    assert attention.shape == volume.shape
    assert float(np.max(attention)) == 1.0
