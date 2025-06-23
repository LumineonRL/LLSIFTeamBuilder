from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass(frozen=True)
class SongData:
    """
    Represents the static, immutable data for a single song, loaded from JSON.
    """
    song_id: str
    title: str
    difficulty: str
    group: str
    attribute: str
    notes: List[Dict[str, Any]] = field(default_factory=list)
