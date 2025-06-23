from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class Gallery:
    """Holds the gallery stat bonuses for a deck."""
    smile: int = 0
    pure: int = 0
    cool: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Serializes the gallery stats to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'Gallery':
        """Creates a Gallery instance from a dictionary."""
        return cls(**data)
