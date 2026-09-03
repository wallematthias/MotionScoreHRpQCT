from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DiscoveryConfig:
    session_regex: str = (
        r"(?i)^(?P<subject>.+?)(?:_(?P<site>DR|DT|KN|RL|RR|TL|TR|KL|KR|RADIUS|TIBIA|KNEE|"
        r"RADIUS_LEFT|RADIUS_RIGHT|TIBIA_LEFT|TIBIA_RIGHT|KNEE_LEFT|KNEE_RIGHT))?"
        r"(?:_STACK(?P<stack>\d+))?_(?P<session>[A-Z][A-Z0-9]*)(?:_(?P<role>.*))?\.aim(?:;\d+)?$"
    )
    default_site: str = "tibia"
    site_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "radius": ["DR", "RADIUS", "RAD"],
            "tibia": ["DT", "TIBIA", "TIB"],
            "knee": ["KN", "KNEE"],
            "radiusleft": ["RL", "RADIUS_LEFT", "RADIUSLEFT", "RADL", "LEFT_RADIUS"],
            "radiusright": ["RR", "RADIUS_RIGHT", "RADIUSRIGHT", "RADR", "RIGHT_RADIUS"],
            "tibialeft": ["TL", "TIBIA_LEFT", "TIBIALEFT", "TIBL", "LEFT_TIBIA"],
            "tibiaright": ["TR", "TIBIA_RIGHT", "TIBIARIGHT", "TIBR", "RIGHT_TIBIA"],
            "kneeleft": ["KL", "KNL", "KNEE_LEFT", "KNEELEFT", "KNEEL", "LEFT_KNEE"],
            "kneeright": ["KR", "KNR", "KNEE_RIGHT", "KNEERIGHT", "KNEER", "RIGHT_KNEE"],
        }
    )
    session_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "T1": ["BASELINE", "BL"],
            "T2": ["FOLLOWUP", "FOLLOWUP1", "FL", "FU"],
        }
    )


@dataclass(slots=True)
class InferenceConfig:
    stackheight: int = 168
    on_incomplete_stack: str = "keep_last"  # keep_last | drop_last | error


@dataclass(slots=True)
class ReviewConfig:
    confidence_threshold: int = 75


@dataclass(slots=True)
class AppConfig:
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
