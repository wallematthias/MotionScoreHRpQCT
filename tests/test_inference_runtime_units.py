from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import motionscore.inference as inf_pkg
import motionscore.inference.model as model_mod
import motionscore.inference.torch_model as torch_mod


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_inference_package_lazy_getattr_modelensemble() -> None:
    cls = inf_pkg.__getattr__("ModelEnsemble")
    assert cls is model_mod.ModelEnsemble
    with pytest.raises(AttributeError):
        inf_pkg.__getattr__("UnknownName")


def test_modelensemble_init_and_load_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        model_mod.ModelEnsemble(model_dir=tmp_path, backend="tensorflow")

    _touch(tmp_path / "DNN_0.pt")
    calls = {"n": 0}

    def _fake_loader(path, device="cpu"):
        calls["n"] += 1
        return SimpleNamespace(path=path, __call__=lambda _x: np.zeros((1, 5), dtype=np.float32))

    monkeypatch.setattr(model_mod, "load_torch_model", _fake_loader)
    ens = model_mod.ModelEnsemble(model_dir=tmp_path, device="cpu")
    ens.load()
    ens.load()
    assert calls["n"] == 1
    assert ens.model_version() == "DNN_0.pt"
    assert len(ens.grad_models()) == 1
    assert ens.model_device() == "cpu"


def test_modelensemble_rejects_bad_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        model_mod.ModelEnsemble(model_dir=tmp_path, device="metal")


def test_modelensemble_predict_missing_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ens = model_mod.ModelEnsemble(model_dir=tmp_path)
    ens.loaded = [model_mod.LoadedModel(path=tmp_path / "DNN_0.pt", model=SimpleNamespace())]
    monkeypatch.setattr(ens, "load", lambda: None)

    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match="PyTorch is required for inference"):
        ens.predict(np.zeros((2, 512, 512, 1), dtype=np.float32))


def test_modelensemble_resolve_any_pt_fallback(tmp_path: Path) -> None:
    _touch(tmp_path / "foo.pt")
    ens = model_mod.ModelEnsemble(model_dir=tmp_path)
    resolved = ens.resolve_model_paths()
    assert [p.name for p in resolved] == ["foo.pt"]


def test_torch_model_require_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match="PyTorch is required for the torch backend"):
        torch_mod._require_torch()


def test_load_torch_model_rejects_non_pt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "DNN_0.h5"
    model_path.write_bytes(b"h5")

    class _FakeModel:
        def eval(self):
            return None

    monkeypatch.setattr(torch_mod, "build_torch_model", lambda: _FakeModel())
    fake_torch = SimpleNamespace(
        load=lambda *_a, **_k: {},
        device=lambda x: x,
    )
    monkeypatch.setattr(torch_mod, "_require_torch", lambda: (fake_torch, SimpleNamespace()))

    with pytest.raises(ValueError, match="Unsupported model format"):
        torch_mod.load_torch_model(model_path, device="cpu")


def test_load_torch_model_rejects_invalid_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "DNN_0.pt"
    model_path.write_bytes(b"pt")

    class _FakeModel:
        def load_state_dict(self, *_a, **_k):
            return None

        def eval(self):
            return None

    fake_torch = SimpleNamespace(
        load=lambda *_a, **_k: "bad-payload",
        device=lambda x: x,
    )
    monkeypatch.setattr(torch_mod, "build_torch_model", lambda: _FakeModel())
    monkeypatch.setattr(torch_mod, "_require_torch", lambda: (fake_torch, SimpleNamespace()))

    with pytest.raises(ValueError, match="Unsupported .pt payload format"):
        torch_mod.load_torch_model(model_path, device="cpu")
