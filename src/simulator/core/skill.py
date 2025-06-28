from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class Skill:
    """Represents the details of a card's special skill."""

    type: Optional[str] = None
    activation: Optional[str] = None
    target: Optional[str] = None
    level: List[int] = field(default_factory=list)
    thresholds: List[int] = field(default_factory=list)
    chances: List[float] = field(default_factory=list)
    values: List[Union[int, float]] = field(default_factory=list)
    durations: List[Union[int, float]] = field(default_factory=list)
