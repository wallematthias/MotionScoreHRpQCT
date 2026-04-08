from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from motionscore.inference.preprocessing import inverse_resize_heatmap
from motionscore.inference.scoring import PredictionResult
from motionscore.io.aim import AimVolume, write_volume_mha

if TYPE_CHECKING:
    from motionscore.inference.model import ModelEnsemble


def _slice_gradcam_torch(model: Any, image_tensor: np.ndarray, class_index: int) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("PyTorch is required for torch Grad-CAM generation.") from exc

    model.zero_grad(set_to_none=True)
    image = torch.from_numpy(np.asarray(image_tensor, dtype=np.float32)).permute(0, 3, 1, 2).contiguous()
    image.requires_grad_(True)

    probs, features = model(image, return_features=True)
    features.retain_grad()

    score = probs[:, class_index].sum()
    score.backward()

    grads = features.grad
    if grads is None:
        return np.zeros((512, 512), dtype=np.float32)

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(features * weights, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    cam = cam[0, 0]

    max_val = torch.max(cam)
    if float(max_val) > 0.0:
        cam = cam / (max_val + 1e-8)
    else:
        cam = torch.zeros_like(cam)
    return cam.detach().cpu().numpy().astype(np.float32)


def generate_gradcam_attention_map(
    aim_volume: AimVolume,
    prediction: PredictionResult,
    ensemble: ModelEnsemble,
    output_path: Path | None = None,
    target_class: int | None = None,
) -> tuple[np.ndarray, Path | None]:
    if prediction.preprocessed_scan is None or not prediction.preprocess_infos:
        raise ValueError(
            "Prediction result does not contain preprocessing cache required for Grad-CAM. "
            "Run predict_scan(..., retain_preprocessed=True) for explain mode."
        )

    depth = aim_volume.data.shape[2]
    if depth != prediction.preprocessed_scan.shape[0]:
        raise ValueError("Prediction preprocessing does not match input volume depth")

    class_index = int(target_class) if target_class is not None else int(prediction.automatic_grade - 1)
    class_index = min(4, max(0, class_index))

    grad_models = ensemble.grad_models()
    attention = np.zeros_like(aim_volume.data, dtype=np.float32)

    for z in range(depth):
        slice_input = prediction.preprocessed_scan[z] / 255.0
        slice_input = np.expand_dims(slice_input, axis=0).astype(np.float32)

        cams = []
        for model in grad_models:
            cams.append(_slice_gradcam_torch(model, slice_input, class_index))
        mean_cam = np.mean(cams, axis=0)

        restored = inverse_resize_heatmap(mean_cam, prediction.preprocess_infos[z])
        attention[:, :, z] = restored

    max_att = float(np.max(attention))
    if max_att > 0:
        attention = attention / max_att

    written_path: Path | None = None
    if output_path is not None:
        written_path = write_volume_mha(
            output_path,
            attention,
            spacing=aim_volume.spacing,
            origin=aim_volume.origin,
            direction=aim_volume.direction,
        )

    return attention, written_path
