from __future__ import annotations

from pathlib import Path
import warnings


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


def build_torch_model():
    torch, nn = _require_torch()

    class MotionScoreTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
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
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    model.eval()
    return model
