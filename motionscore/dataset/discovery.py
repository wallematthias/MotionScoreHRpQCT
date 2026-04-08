from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from motionscore.config import DiscoveryConfig
from motionscore.dataset.layout import PIPELINE_NAME
from motionscore.dataset.models import RawSession

_AIM_WITH_OPTIONAL_VERSION_RE = re.compile(r"(?i)\.aim(?:;\d+)?$")
_EXCLUDE_KEYWORDS = ("mask", "trab", "cort", "full", "regmask", "seg", "roi")
_HEADER_SITE_CODE_MAP = {
    "20": "radius_left",
    "21": "radius_right",
    "38": "tibia_left",
    "29": "tibia_right",
}


def _is_aim_file(path: Path) -> bool:
    return path.is_file() and _AIM_WITH_OPTIONAL_VERSION_RE.search(path.name) is not None


def _strip_aim_suffix(name: str) -> str:
    return _AIM_WITH_OPTIONAL_VERSION_RE.sub("", name)


def _is_pipeline_managed_copy(path: Path, root: Path) -> bool:
    try:
        rel_parts = [p.lower() for p in path.relative_to(root).parts]
    except ValueError:
        return False

    for i in range(len(rel_parts) - 1):
        if rel_parts[i] == "derivatives" and rel_parts[i + 1] == PIPELINE_NAME.lower():
            return True
    return False


def _normalize_site(site_token: str | None, cfg: DiscoveryConfig) -> str | None:
    if not site_token:
        return None
    token = site_token.strip().upper()
    for canonical, aliases in cfg.site_aliases.items():
        valid = {canonical.upper(), *(a.upper() for a in aliases)}
        if token in valid:
            return canonical.lower()
    return site_token.strip().lower()


def _normalize_session(session_token: str, cfg: DiscoveryConfig) -> str:
    token = session_token.strip()
    token_upper = token.upper()

    followup = re.fullmatch(r"(?:FL|FU|FOLLOWUP)(\d+)", token_upper)
    if followup:
        return f"T{int(followup.group(1)) + 1}"
    if re.fullmatch(r"(?:BL|BASELINE)(?:1+)?", token_upper):
        return "T1"

    for canonical, aliases in cfg.session_aliases.items():
        valid = {canonical.upper(), *(a.upper() for a in aliases)}
        if token_upper in valid:
            return canonical
    return token


def _role_from_name(path: Path) -> str:
    stem = _strip_aim_suffix(path.name).upper()
    if any(
        k in stem
        for k in (
            "TRAB_MASK",
            "_TRAB",
            "CORT_MASK",
            "_CORT",
            "FULL_MASK",
            "_FULL",
            "REGMASK",
            "_REG",
            "_SEG",
        )
    ):
        return "derived"
    if re.search(r"(?i)_ROI[0-9A-Z]+$", stem):
        return "derived"
    if re.search(r"(?i)_MASK[0-9A-Z]+$", stem):
        return "derived"
    return "image"


def _extract_stack_index(path: Path) -> int | None:
    stem = _strip_aim_suffix(path.name)
    m = re.search(r"(?i)(?:^|_)STACK[_-]?(\d+)(?:_|$)", stem)
    return int(m.group(1)) if m else None


def _extract_by_regex(path: Path, cfg: DiscoveryConfig) -> tuple[str, str, str | None, int | None, str]:
    m = re.search(cfg.session_regex, path.name)
    if not m:
        raise ValueError("filename did not match discovery regex")

    groups = m.groupdict()
    subject = groups.get("subject")
    session = groups.get("session")
    site_token = groups.get("site")
    role_token = groups.get("role")
    stack_text = groups.get("stack")

    if not subject or not session:
        raise ValueError("regex must capture subject and session")

    site = _normalize_site(site_token, cfg)
    role = "image" if not role_token else _role_from_name(Path(f"x_{role_token}.aim"))

    return subject, _normalize_session(session, cfg), site, int(stack_text) if stack_text else None, role


def _extract_default(path: Path, cfg: DiscoveryConfig) -> tuple[str, str, str, int | None, str]:
    stem = _strip_aim_suffix(path.name)
    stack_index = _extract_stack_index(path)
    stem = re.sub(r"(?i)_STACK[_-]?\d+", "", stem)

    role = _role_from_name(path)
    if role != "image":
        stem = re.sub(
            r"(?i)_TRAB_MASK$|_CORT_MASK$|_FULL_MASK$|_REGMASK$|_SEG$|_TRAB$|_CORT$|_FULL$",
            "",
            stem,
        )
        stem = re.sub(r"(?i)_ROI[0-9A-Z]+$|_MASK[0-9A-Z]+$", "", stem)

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 3:
        raise ValueError(f"Cannot parse filename: {path.name}")

    session = _normalize_session(parts[-1], cfg)
    site = _normalize_site(parts[-2], cfg) or cfg.default_site.lower()
    subject = "_".join(parts[:-2])
    if not subject:
        raise ValueError(f"Cannot infer subject from filename: {path.name}")

    return subject, session, site, stack_index, role


def _parse_processing_log_to_dict(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def _processing_log_as_text(meta: dict) -> str:
    log = meta.get("processing_log_raw", meta.get("processing_log", ""))
    if isinstance(log, dict):
        return "\n".join(f"{k}: {v}" for k, v in log.items())
    return str(log)


def _infer_role_from_processing_log(processing_log: str) -> str | None:
    text = processing_log.upper()
    # Strong raw-image signals from AIM header history.
    if "ORIG-ISQ" in text:
        return "image"
    if re.search(r"ORIGINAL FILE\s+.*\.ISQ", text):
        return "image"
    if "ISQ_TO_AIM" in text:
        return "image"

    # Common derived/mask pipeline signals.
    if "AIMPEEL" in text:
        return "derived"
    if "CREATEAIM" in text and "MASK" in text:
        return "derived"
    if "SEG" in text and "CREATED BY" in text:
        return "derived"

    return None


def _role_from_header(path: Path) -> str | None:
    try:
        import py_aimio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError("py_aimio unavailable for header role detection") from exc

    try:
        meta = dict(py_aimio.aim_info(str(path)))
    except Exception as exc:
        raise ValueError(f"failed to read AIM header role for {path.name}") from exc
    processing_log = _processing_log_as_text(meta)
    return _infer_role_from_processing_log(processing_log)


def _extract_from_header(path: Path, cfg: DiscoveryConfig) -> tuple[str, str, str, int | None, str]:
    try:
        import py_aimio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError("py_aimio unavailable for header fallback") from exc

    try:
        meta = dict(py_aimio.aim_info(str(path)))
    except Exception as exc:
        raise ValueError(f"failed to read AIM header for {path.name}") from exc
    processing_log = _processing_log_as_text(meta)
    log_dict = _parse_processing_log_to_dict(processing_log)

    subject = str(log_dict.get("Index Patient", "")).strip()
    if not subject:
        raise ValueError(f"Header fallback missing Index Patient for {path.name}")

    measurement = str(log_dict.get("Index Measurement", "")).strip()
    session = _normalize_session(f"M{measurement}" if measurement else "T1", cfg)

    site_raw = str(log_dict.get("Site", "")).strip()
    if site_raw in _HEADER_SITE_CODE_MAP:
        site = _HEADER_SITE_CODE_MAP[site_raw]
    else:
        site = _normalize_site(site_raw, cfg) or cfg.default_site.lower()

    stack_index = _extract_stack_index(path)
    role = _infer_role_from_processing_log(processing_log) or _role_from_name(path)
    return subject, session, site, stack_index, role


def _compute_output_rel_dir(root: Path, raw_image_path: Path) -> Path:
    try:
        rel_parent = raw_image_path.relative_to(root).parent
    except ValueError:
        rel_parent = Path(".")

    if rel_parent == Path("."):
        return Path(_strip_aim_suffix(raw_image_path.name))
    return rel_parent


def discover_raw_sessions(
    root: str | Path,
    cfg: DiscoveryConfig,
    force_header_discovery: bool = False,
) -> list[RawSession]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Discovery root does not exist: {root}")

    grouped: dict[tuple[str, str, str, int | None], list[tuple[Path, str]]] = defaultdict(list)

    for path in root.rglob("*"):
        if not _is_aim_file(path):
            continue
        if _is_pipeline_managed_copy(path, root):
            continue

        try:
            if force_header_discovery:
                subject, session, site, stack_index, role = _extract_from_header(path, cfg)
            else:
                try:
                    subject, session, site, stack_index, role = _extract_by_regex(path, cfg)
                    site = site or cfg.default_site.lower()
                except ValueError:
                    try:
                        subject, session, site, stack_index, role = _extract_default(path, cfg)
                    except ValueError:
                        subject, session, site, stack_index, role = _extract_from_header(path, cfg)
        except ValueError:
            continue

        # Header-based role classification (raw vs mask) is primary when available.
        try:
            header_role = _role_from_header(path)
            if header_role:
                role = header_role
        except ValueError:
            pass

        grouped[(subject, site, session, stack_index)].append((path, role))

    sessions: list[RawSession] = []
    for (subject, site, session, stack_index), items in sorted(grouped.items()):
        image_candidates = [
            p for p, role in items if role == "image" and not any(k in p.name.lower() for k in _EXCLUDE_KEYWORDS)
        ]

        if len(image_candidates) == 0:
            continue
        if len(image_candidates) > 1:
            raise ValueError(
                "Multiple ambiguous raw image AIMs found for "
                f"{subject}/{site}/{session}: {', '.join(str(p) for p in image_candidates)}"
            )

        raw_session = RawSession(
            subject_id=subject,
            site=site,
            session_id=session,
            raw_image_path=image_candidates[0],
            stack_index=stack_index,
            output_rel_dir=_compute_output_rel_dir(root, image_candidates[0]),
        )
        raw_session.validate()
        sessions.append(raw_session)

    return sessions
