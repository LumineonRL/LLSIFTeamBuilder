from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LeaderSkill:
    """Represents the details of a card's leader skill."""
    attribute: Optional[str] = None
    secondary_attribute: Optional[str] = None
    value: float = 0.0
