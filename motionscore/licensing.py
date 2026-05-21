from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from motionscore.model_registry import register_model_profile


DEFAULT_MODEL_CATALOG_URL = (
    "https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json"
)


def load_model_catalog(source: str | Path | None = None) -> dict[str, Any]:
    raw_source = str(source or os.environ.get("MOTIONSCORE_MODEL_CATALOG_URL") or DEFAULT_MODEL_CATALOG_URL).strip()
    if not raw_source:
        raise ValueError("model catalog source must not be empty")
    if raw_source.startswith(("http://", "https://")):
        with urllib.request.urlopen(raw_source, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(raw_source).expanduser().read_text(encoding="utf-8"))


def download_and_register_model(
    *,
    model_id: str,
    model_root: str | Path,
    catalog_source: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    catalog = load_model_catalog(catalog_source)
    model = _find_model(catalog, model_id)
    root = Path(model_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / _safe_component(str(model.get("model_id", model_id)))
    if target_dir.exists() and any(target_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Model directory already exists: {target_dir}. Use --overwrite to replace it.")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = _download_to_temp(str(model["url"]))
    try:
        sha256 = str(model.get("sha256", "") or "").strip().lower()
        if sha256:
            actual = _sha256(downloaded)
            if actual.lower() != sha256:
                raise ValueError(f"Checksum mismatch for {downloaded.name}: expected {sha256}, got {actual}")
        _extract_or_copy_model(downloaded, target_dir)
    finally:
        downloaded.unlink(missing_ok=True)

    registry_path, entry = register_model_profile(
        model_root=root,
        model_id=str(model.get("model_id", model_id)),
        model_dir=target_dir,
        display_name=str(model.get("display_name", model_id)),
        domain=str(model.get("domain", "motion-score")),
        version=str(model.get("version", "")),
        description=str(model.get("description", "")),
        source_model_id=str(model.get("source_model_id", "")),
        make_default=bool(model.get("make_default", True)),
    )
    _write_download_receipt(target_dir, model, catalog_source)
    return registry_path, entry


def _safe_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    return safe.strip(".-") or "model"


def _find_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    requested = str(model_id or "").strip()
    models = catalog.get("models", [])
    if not isinstance(models, list):
        raise ValueError("model catalog must contain a 'models' list")
    if not requested:
        requested = str(catalog.get("default_model_id", "") or "").strip()
    if not requested:
        raise ValueError("model_id is required because the catalog has no default_model_id")
    for raw in models:
        if isinstance(raw, dict) and str(raw.get("model_id", "")).strip() == requested:
            if not str(raw.get("url", "")).strip():
                raise ValueError(f"model catalog entry '{requested}' is missing url")
            return dict(raw)
    raise KeyError(f"model_id '{requested}' not found in model catalog")


def _download_to_temp(url: str) -> Path:
    parsed_name = Path(urllib.parse.urlparse(url).path).name
    suffix = ".tar.gz" if parsed_name.lower().endswith(".tar.gz") else Path(parsed_name).suffix
    handle, name = tempfile.mkstemp(prefix="motionscore-model-", suffix=suffix)
    os.close(handle)
    out = Path(name)
    try:
        with urllib.request.urlopen(url, timeout=300) as response, out.open("wb") as f:
            shutil.copyfileobj(response, f)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out


def _extract_or_copy_model(archive_path: Path, target_dir: Path) -> None:
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            _safe_extract_zip(zf, target_dir)
        _flatten_single_model_subdir(target_dir)
        return
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path) as tf:
            _safe_extract_tar(tf, target_dir)
        _flatten_single_model_subdir(target_dir)
        return
    if lower.endswith(".pt"):
        shutil.copy2(archive_path, target_dir / archive_path.name)
        return
    raise ValueError(f"Unsupported model download format: {archive_path.name}")


def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    target = target_dir.resolve()
    for info in zf.infolist():
        member_path = (target / info.filename).resolve()
        if target != member_path and target not in member_path.parents:
            raise ValueError(f"Unsafe archive path: {info.filename}")
    zf.extractall(target)


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    target = target_dir.resolve()
    for member in tf.getmembers():
        member_path = (target / member.name).resolve()
        if target != member_path and target not in member_path.parents:
            raise ValueError(f"Unsafe archive path: {member.name}")
    tf.extractall(target)


def _flatten_single_model_subdir(target_dir: Path) -> None:
    if list(target_dir.glob("DNN_*.pt")):
        return
    children = [p for p in target_dir.iterdir() if p.is_dir()]
    files = [p for p in target_dir.iterdir() if p.is_file()]
    if len(children) != 1 or files:
        return
    child = children[0]
    if not list(child.glob("DNN_*.pt")):
        return
    temp = target_dir / ".motionscore_extract"
    child.rename(temp)
    for item in temp.iterdir():
        item.rename(target_dir / item.name)
    temp.rmdir()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_download_receipt(
    target_dir: Path,
    model: dict[str, Any],
    catalog_source: str | Path | None,
) -> None:
    payload = {
        "downloaded_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_id": model.get("model_id", ""),
        "version": model.get("version", ""),
        "catalog_source": str(catalog_source or os.environ.get("MOTIONSCORE_MODEL_CATALOG_URL") or DEFAULT_MODEL_CATALOG_URL),
    }
    (target_dir / "motionscore_download_receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
