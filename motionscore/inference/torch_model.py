from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np


def _require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h5py is required for .h5 model loading/conversion. Install with pip install -e '.[h5]'."
        ) from exc
    return h5py


def _require_torch():
    try:
        # Torch can emit a noisy warning in mixed environments when NumPy C-API
        # features are unavailable; inference has a runtime fallback for that.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Failed to initialize NumPy: _ARRAY_API not found.*",
                category=UserWarning,
            )
            import torch
            import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the torch backend. Install with `pip install -e '.[torch]'`."
        ) from exc
    return torch, nn


def _as_tensor(arr: np.ndarray):
    torch, _ = _require_torch()
    arr_np = np.asarray(arr, dtype=np.float32)
    try:
        return torch.from_numpy(arr_np)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        return torch.tensor(arr_np.tolist(), dtype=torch.float32)


def build_torch_model():
    torch, nn = _require_torch()

    class MotionScoreTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Keras source architecture:
            # Conv(1->16,k3,valid,LeakyReLU) x5 with MaxPool after first four
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
            self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
            self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
            self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.leaky = nn.LeakyReLU(negative_slope=0.3, inplace=False)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 5)

        def forward(self, x, *, return_features: bool = False):
            x = self.leaky(self.conv1(x))
            x = self.pool(x)

            x = self.leaky(self.conv2(x))
            x = self.pool(x)

            x = self.leaky(self.conv3(x))
            x = self.pool(x)

            x = self.leaky(self.conv4(x))
            x = self.pool(x)

            features = self.leaky(self.conv5(x))
            x = self.gap(features).view(features.shape[0], -1)
            x = torch.relu(self.fc1(x))
            logits = self.fc2(x)
            probs = torch.softmax(logits, dim=1)

            if return_features:
                return probs, features
            return probs

    return MotionScoreTorchNet()


def _read_model_config(h5_file) -> dict:
    raw = h5_file.attrs.get("model_config")
    if raw is None:
        raise ValueError("model_config not found in .h5 model")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        import json

        return json.loads(raw)
    raise ValueError(f"Unsupported model_config type: {type(raw)}")


def _find_weight_group(layer_root):
    h5py = _require_h5py()
    if "kernel:0" in layer_root and "bias:0" in layer_root:
        return layer_root
    for key in layer_root.keys():
        candidate = layer_root[key]
        if isinstance(candidate, h5py.Group) and "kernel:0" in candidate and "bias:0" in candidate:
            return candidate
    raise KeyError(f"Could not locate kernel/bias datasets under layer group: {layer_root.name}")


def _read_keras_layer_from_file(h5_file, layer_name: str) -> tuple[np.ndarray, np.ndarray]:
    mw = h5_file["model_weights"]
    if layer_name not in mw:
        raise KeyError(f"Layer '{layer_name}' not found in model_weights")
    layer_root = mw[layer_name]
    weight_group = _find_weight_group(layer_root)
    kernel = np.asarray(weight_group["kernel:0"], dtype=np.float32)
    bias = np.asarray(weight_group["bias:0"], dtype=np.float32)
    return kernel, bias


def load_keras_h5_weights_into_torch_model(model, h5_path: Path) -> Any:
    h5py = _require_h5py()
    h5_path = Path(h5_path).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        model_cfg = _read_model_config(f)
        layers = model_cfg.get("config", {}).get("layers", [])
        conv_names = [l.get("config", {}).get("name") for l in layers if l.get("class_name") == "Conv2D"]
        dense_names = [l.get("config", {}).get("name") for l in layers if l.get("class_name") == "Dense"]

        if len(conv_names) != 5:
            raise ValueError(f"Expected 5 Conv2D layers, found {len(conv_names)} in {h5_path.name}")
        if len(dense_names) != 2:
            raise ValueError(f"Expected 2 Dense layers, found {len(dense_names)} in {h5_path.name}")

        # Conv2D in Keras: [H, W, in_c, out_c]
        # Conv2D in torch: [out_c, in_c, H, W]
        conv_modules = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
        for layer_name, module in zip(conv_names, conv_modules):
            kernel, bias = _read_keras_layer_from_file(f, layer_name)
            weight = np.transpose(kernel, (3, 2, 0, 1))
            module.weight.data.copy_(_as_tensor(weight))
            module.bias.data.copy_(_as_tensor(bias))

        # Dense in Keras: [in, out]
        # Linear in torch: [out, in]
        dense_kernel, dense_bias = _read_keras_layer_from_file(f, dense_names[0])
        model.fc1.weight.data.copy_(_as_tensor(dense_kernel.T))
        model.fc1.bias.data.copy_(_as_tensor(dense_bias))

        dense2_kernel, dense2_bias = _read_keras_layer_from_file(f, dense_names[1])
        model.fc2.weight.data.copy_(_as_tensor(dense2_kernel.T))
        model.fc2.bias.data.copy_(_as_tensor(dense2_bias))

    return model


def load_torch_model(model_path: Path):
    model_path = Path(model_path).resolve()
    model = build_torch_model()
    torch, _ = _require_torch()

    if model_path.suffix.lower() == ".pt":
        payload = torch.load(str(model_path), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported .pt payload format: {type(payload)}")
        model.load_state_dict(state_dict, strict=True)
    elif model_path.suffix.lower() == ".h5":
        load_keras_h5_weights_into_torch_model(model, model_path)
    else:
        raise ValueError(f"Unsupported model format for torch backend: {model_path}")

    model.eval()
    return model


def save_torch_from_h5(h5_path: Path, pt_path: Path, overwrite: bool = False) -> Path:
    h5_path = Path(h5_path).resolve()
    pt_path = Path(pt_path).resolve()
    if pt_path.exists() and not overwrite:
        return pt_path

    model = build_torch_model()
    load_keras_h5_weights_into_torch_model(model, h5_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch, _ = _require_torch()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "source_h5": str(h5_path),
        },
        str(pt_path),
    )
    return pt_path
