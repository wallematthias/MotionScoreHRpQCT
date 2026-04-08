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
            "radius_left": ["RL", "RADIUS_LEFT", "RADL", "LEFT_RADIUS"],
            "radius_right": ["RR", "RADIUS_RIGHT", "RADR", "RIGHT_RADIUS"],
            "tibia_left": ["TL", "TIBIA_LEFT", "TIBL", "LEFT_TIBIA"],
            "tibia_right": ["TR", "TIBIA_RIGHT", "TIBR", "RIGHT_TIBIA"],
            "knee_left": ["KL", "KNL", "KNEE_LEFT", "KNEEL", "LEFT_KNEE"],
            "knee_right": ["KR", "KNR", "KNEE_RIGHT", "KNEER", "RIGHT_KNEE"],
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
