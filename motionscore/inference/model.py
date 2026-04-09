from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motionscore.inference.torch_model import load_torch_model
from motionscore.model_registry import resolve_model_dir


@dataclass(slots=True)
class LoadedModel:
    path: Path
    model: Any
    grad_model: Any | None = None


class ModelEnsemble:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_root: str | Path | None = None,
        model_id: str | None = None,
        backend: str = "torch",
        device: str = "auto",
    ):
        backend_norm = str(backend).strip().lower()
        if backend_norm != "torch":
            raise ValueError("Only torch backend is supported")
        device_norm = str(device).strip().lower()
        if device_norm not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, mps, cuda")

        self.model_dir = Path(model_dir).resolve() if model_dir else None
        self.model_root = Path(model_root).resolve() if model_root else None
        self.model_id = str(model_id).strip() if model_id is not None else "base-v1"
        self.backend = "torch"
        self.requested_device = device_norm
        self.runtime_device = "cpu"
        self.resolved_model_dir: Path | None = None
        self.resolved_profile: dict[str, Any] | None = None
        self.loaded: list[LoadedModel] = []

    def resolve_model_paths(self) -> list[Path]:
        if self.model_dir is not None:
            candidate_dir = self.model_dir.resolve()
            self.resolved_profile = None
        else:
            root = self.model_root.resolve() if self.model_root is not None else (Path.cwd() / "models").resolve()
            candidate_dir, profile = resolve_model_dir(root, model_id=self.model_id)
            self.resolved_profile = profile

        self.resolved_model_dir = candidate_dir
        dnn = sorted(candidate_dir.glob("DNN_*.pt"))
        if dnn:
            return dnn

        any_models = sorted(candidate_dir.glob("*.pt"))
        if any_models:
            return any_models

        raise FileNotFoundError(f"No .pt model files found in: {candidate_dir}")

    def _resolve_torch_device(self) -> str:
        if self.requested_device == "cpu":
            return "cpu"

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for inference. Install with pip install -e '.[torch]'."
            ) from exc

        if self.requested_device == "cuda":
            if not bool(torch.cuda.is_available()):
                raise RuntimeError("Requested device=cuda but CUDA is not available.")
            return "cuda"

        if self.requested_device == "mps":
            has_mps = bool(getattr(torch.backends, "mps", None))
            if not has_mps or not bool(torch.backends.mps.is_available()):
                raise RuntimeError("Requested device=mps but MPS is not available.")
            return "mps"

        # auto
        has_mps = bool(getattr(torch.backends, "mps", None))
        if has_mps and bool(torch.backends.mps.is_available()):
            return "mps"
        if bool(torch.cuda.is_available()):
            return "cuda"
        return "cpu"

    def _load_torch(self, model_paths: list[Path]) -> None:
        resolved_device = self._resolve_torch_device()
        loaded: list[LoadedModel] = []
        for path in model_paths:
            model = load_torch_model(path, device=resolved_device)
            loaded.append(LoadedModel(path=path, model=model, grad_model=None))
        self.runtime_device = resolved_device
        self.loaded = loaded

    def load(self) -> None:
        if self.loaded:
            return
        model_paths = self.resolve_model_paths()
        self._load_torch(model_paths)

    def model_version(self) -> str:
        self.load()
        names = [m.path.name for m in self.loaded]
        return "+".join(names)

    def model_identity(self) -> str:
        self.load()
        if self.resolved_profile:
            model_id = str(self.resolved_profile.get("model_id", "")).strip()
            version = str(self.resolved_profile.get("version", "")).strip()
            model_prefix = f"{model_id}@{version}" if version else model_id
            return f"{model_prefix}:{self.model_version()}" if model_prefix else self.model_version()
        if self.model_dir is not None:
            return f"{self.model_dir.name}:{self.model_version()}"
        return f"{self.model_id}:{self.model_version()}"

    def resolved_model_id(self) -> str:
        self.load()
        if self.resolved_profile:
            return str(self.resolved_profile.get("model_id", "")).strip() or self.model_id
        if self.model_dir is not None:
            return self.model_dir.name
        return self.model_id

    def model_device(self) -> str:
        self.load()
        return self.runtime_device

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.load()

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for inference. Install with pip install -e '.[torch]'."
            ) from exc

        batch_np = np.asarray(batch, dtype=np.float32)
        try:
            x = torch.from_numpy(batch_np)
        except RuntimeError as exc:
            if "Numpy is not available" not in str(exc):
                raise
            x = torch.tensor(batch_np.tolist(), dtype=torch.float32)

        x = x.permute(0, 3, 1, 2).contiguous().to(self.runtime_device)
        predictions = np.zeros((len(self.loaded), batch.shape[0], 5), dtype=np.float32)
        with torch.no_grad():
            for i, loaded in enumerate(self.loaded):
                out_cpu = loaded.model(x).detach().cpu()
                try:
                    out = out_cpu.numpy().astype(np.float32)
                except RuntimeError as exc:
                    if "Numpy is not available" not in str(exc):
                        raise
                    out = np.asarray(out_cpu.tolist(), dtype=np.float32)
                predictions[i] = out
        return predictions

    def grad_models(self) -> list[Any]:
        self.load()
        return [m.model for m in self.loaded]
