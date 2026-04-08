from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
from typing import Any
import warnings

import numpy as np

from motionscore.inference.torch_model import load_torch_model, save_torch_from_h5


@dataclass(slots=True)
class LoadedModel:
    path: Path
    model: Any
    grad_model: Any | None = None


class ModelEnsemble:
    _tf_deprecation_warned = False

    def __init__(
        self,
        model_dir: str | Path | None = None,
        backend: str = "tensorflow",
    ):
        backend_norm = str(backend).strip().lower()
        if backend_norm not in {"tensorflow", "torch"}:
            raise ValueError("backend must be one of: tensorflow, torch")

        self.model_dir = Path(model_dir).resolve() if model_dir else None
        self.backend = backend_norm
        self.loaded: list[LoadedModel] = []
        self._warn_tensorflow_deprecation_if_needed()

    def _warn_tensorflow_deprecation_if_needed(self) -> None:
        if self.backend != "tensorflow" or ModelEnsemble._tf_deprecation_warned:
            return
        warnings.warn(
            "TensorFlow backend is deprecated and will be removed in a future release. "
            "Please migrate to '--backend torch'.",
            FutureWarning,
            stacklevel=3,
        )
        ModelEnsemble._tf_deprecation_warned = True

    @staticmethod
    def _candidate_model_dirs() -> list[Path]:
        package_root = Path(__file__).resolve().parents[1]
        repo_root = Path(__file__).resolve().parents[2]
        return [
            package_root / "models",
            repo_root / "models",
            Path.cwd() / "models",
        ]

    def _resolve_candidate_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        if self.model_dir is not None:
            dirs.append(self.model_dir)
        dirs.extend(self._candidate_model_dirs())

        seen: set[Path] = set()
        unique_dirs: list[Path] = []
        for d in dirs:
            d = d.resolve()
            if d in seen:
                continue
            seen.add(d)
            unique_dirs.append(d)
        return unique_dirs

    def resolve_model_paths(self) -> list[Path]:
        unique_dirs = self._resolve_candidate_dirs()

        if self.backend == "tensorflow":
            suffix_order = [".h5"]
        else:
            # Torch backend supports native .pt and converted-on-load .h5.
            suffix_order = [".pt", ".h5"]

        for d in unique_dirs:
            if not d.exists():
                continue
            for suffix in suffix_order:
                dnn = sorted(d.glob(f"DNN_*{suffix}"))
                if dnn:
                    return dnn
                any_models = sorted(d.glob(f"*{suffix}"))
                if any_models:
                    return any_models

        searched = ", ".join(str(d) for d in unique_dirs)
        raise FileNotFoundError(f"No model files found. Searched: {searched}")

    def resolve_h5_paths(self) -> list[Path]:
        unique_dirs = self._resolve_candidate_dirs()
        for d in unique_dirs:
            if not d.exists():
                continue
            dnn = sorted(d.glob("DNN_*.h5"))
            if dnn:
                return dnn
            any_h5 = sorted(d.glob("*.h5"))
            if any_h5:
                return any_h5
        searched = ", ".join(str(d) for d in unique_dirs)
        raise FileNotFoundError(f"No .h5 model files found. Searched: {searched}")

    @staticmethod
    def _resolve_last_conv_layer(model):
        try:
            from tensorflow.keras.layers import Conv2D
        except Exception as exc:
            raise RuntimeError("TensorFlow is required for model loading") from exc

        conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
        if not conv_layers:
            raise ValueError("Model does not include a Conv2D layer required for Grad-CAM")
        return conv_layers[-1]

    @staticmethod
    def _assert_tensorflow_runtime() -> None:
        """Fail fast with actionable diagnostics before in-process TensorFlow import.

        On some macOS/Intel builds TensorFlow requires AVX instructions and can abort
        the interpreter hard if they are unavailable. We gate import using CPU feature
        checks to avoid abrupt process exits.
        """
        machine = platform.machine().lower()
        # AVX is an x86/x86_64 feature; arm64 runtimes cannot satisfy it.
        if machine in {"arm64", "aarch64"}:
            raise RuntimeError(
                "TensorFlow cannot run in this runtime because AVX CPU instructions are unavailable "
                "(non-x86 architecture detected)."
            )

        if machine not in {"x86_64", "amd64", "i386", "i686"}:
            return

        # macOS: inspect sysctl CPU feature flags.
        if platform.system().lower() == "darwin":
            features = []
            for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
                try:
                    probe = subprocess.run(
                        ["sysctl", "-n", key],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=2,
                    )
                    if probe.returncode == 0 and probe.stdout:
                        features.append(probe.stdout.strip().upper())
                except Exception:
                    continue
            feature_blob = " ".join(features)
            if feature_blob and "AVX" not in feature_blob:
                raise RuntimeError(
                    "TensorFlow cannot run on this machine because AVX CPU instructions are unavailable "
                    "in the current Python runtime."
                )
            return

        # Linux: inspect /proc/cpuinfo flags.
        if platform.system().lower() == "linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    cpuinfo = f.read().lower()
                if " avx" not in cpuinfo:
                    raise RuntimeError(
                        "TensorFlow cannot run on this machine because AVX CPU instructions are unavailable."
                    )
            except FileNotFoundError:
                pass
            return

        # Other OSes: best-effort, no hard block.
        return

    @staticmethod
    def _debug_cpu_features() -> str:
        try:
            machine = platform.machine()
            return f"machine={machine}"
        except Exception:
            return "machine=unknown"

    def _load_tensorflow(self, model_paths: list[Path]) -> None:
        try:
            self._assert_tensorflow_runtime()
        except Exception as exc:
            raise RuntimeError(f"{exc} ({self._debug_cpu_features()})") from exc

        try:
            from tensorflow.keras.layers import LeakyReLU
            from tensorflow.keras.models import Model, load_model
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow is required for inference. Install with pip install -e '.[tensorflow]'."
            ) from exc

        loaded: list[LoadedModel] = []
        for i, path in enumerate(model_paths):
            model = load_model(path, compile=False, custom_objects={"LeakyReLU": LeakyReLU()})
            model._name = f"model{i}"
            conv_layer = self._resolve_last_conv_layer(model)
            grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
            loaded.append(LoadedModel(path=path, model=model, grad_model=grad_model))
        self.loaded = loaded

    def _load_torch(self, model_paths: list[Path]) -> None:
        loaded: list[LoadedModel] = []
        for path in model_paths:
            model = load_torch_model(path)
            loaded.append(LoadedModel(path=path, model=model, grad_model=None))
        self.loaded = loaded

    def load(self) -> None:
        if self.loaded:
            return
        model_paths = self.resolve_model_paths()
        if self.backend == "tensorflow":
            self._load_tensorflow(model_paths)
        else:
            self._load_torch(model_paths)

    def model_version(self) -> str:
        self.load()
        names = [m.path.name for m in self.loaded]
        return "+".join(names)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.load()

        if self.backend == "tensorflow":
            predictions = np.zeros((len(self.loaded), batch.shape[0], 5), dtype=np.float32)
            for i, loaded in enumerate(self.loaded):
                predictions[i] = loaded.model.predict(batch, verbose=0)
            return predictions

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for torch backend inference. Install with pip install -e '.[torch]'."
            ) from exc

        batch_np = np.asarray(batch, dtype=np.float32)
        try:
            x = torch.from_numpy(batch_np)
        except RuntimeError as exc:
            # Some torch builds cannot use NumPy bridge APIs; fallback via Python lists.
            if "Numpy is not available" not in str(exc):
                raise
            x = torch.tensor(batch_np.tolist(), dtype=torch.float32)

        x = x.permute(0, 3, 1, 2).contiguous()
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
        if self.backend == "tensorflow":
            return [m.grad_model for m in self.loaded]
        return [m.model for m in self.loaded]

    def convert_h5_to_torch(
        self,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        h5_paths = self.resolve_h5_paths()
        target_dir = Path(output_dir).resolve() if output_dir else h5_paths[0].resolve().parent
        target_dir.mkdir(parents=True, exist_ok=True)

        converted: list[Path] = []
        for h5_path in h5_paths:
            pt_path = target_dir / f"{h5_path.stem}.pt"
            converted.append(save_torch_from_h5(h5_path=h5_path, pt_path=pt_path, overwrite=overwrite))
        return converted
