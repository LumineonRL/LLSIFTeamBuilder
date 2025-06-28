from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Stats:
    """Holds the core stats for a card in a specific idolization state."""

    smile: int = 0
    pure: int = 0
    cool: int = 0
    sis_base: int = 1
    sis_max: int = 1
    image: Optional[str] = None
