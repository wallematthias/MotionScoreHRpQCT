from pathlib import Path

from motionscore.inference.model import ModelEnsemble


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_resolve_model_paths_tensorflow_prefers_h5(tmp_path: Path) -> None:
    _touch(tmp_path / "DNN_0.h5")
    _touch(tmp_path / "DNN_0.pt")
    ensemble = ModelEnsemble(model_dir=tmp_path, backend="tensorflow")
    resolved = ensemble.resolve_model_paths()
    assert resolved
    assert resolved[0].suffix == ".h5"


def test_resolve_model_paths_torch_prefers_pt(tmp_path: Path) -> None:
    _touch(tmp_path / "DNN_0.h5")
    _touch(tmp_path / "DNN_0.pt")
    ensemble = ModelEnsemble(model_dir=tmp_path, backend="torch")
    resolved = ensemble.resolve_model_paths()
    assert resolved
    assert resolved[0].suffix == ".pt"
