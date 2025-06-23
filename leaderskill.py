from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LeaderSkill:
    """Represents the details of a card's leader skill, including any extra components."""
    attribute: Optional[str] = None
    secondary_attribute: Optional[str] = None
    value: float = 0.0
    extra_attribute: Optional[str] = None
    extra_target: Optional[str] = None
    extra_value: float = 0.0
