from pathlib import Path

from motionscore.inference.model import ModelEnsemble


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_resolve_model_paths_torch_prefers_pt(tmp_path: Path) -> None:
    _touch(tmp_path / "DNN_0.pt")
    _touch(tmp_path / "DNN_1.pt")
    ensemble = ModelEnsemble(model_dir=tmp_path)
    resolved = ensemble.resolve_model_paths()
    assert [p.name for p in resolved] == ["DNN_0.pt", "DNN_1.pt"]


def test_resolve_model_paths_requires_pt_models(tmp_path: Path) -> None:
    _touch(tmp_path / "DNN_0.h5")
    ensemble = ModelEnsemble(model_dir=tmp_path)
    try:
        ensemble.resolve_model_paths()
        raise AssertionError("Expected FileNotFoundError when no .pt models are available")
    except FileNotFoundError:
        pass
