from dataclasses import dataclass

@dataclass(frozen=True)
class Note:
    """
    Represents a single immutable note in a song's beatmap.
    """
    start_time: float
    end_time: float
    position: int
    is_star: bool
    is_swing: bool
