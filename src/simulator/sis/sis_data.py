from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SISData:
    """Represents the static, immutable data for a single SIS."""

    id: int
    name: str
    effect: str
    slots: int
    attribute: str
    group: Optional[str] = None
    equip_restriction: Optional[str] = None
    target: Optional[str] = None
    value: float = 0.0
