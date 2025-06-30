from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

from src.simulator import Team

class BuildPhase(IntEnum):
    """Defines the distinct phases of the team-building process."""

    CARD_SELECTION = 0
    ACCESSORY_ASSIGNMENT = 1
    SIS_ASSIGNMENT = 2
    GUEST_SELECTION = 3
    SCORE_SIMULATION = 4


@dataclass
class GameState:
    """Encapsulates the dynamic state of the game environment."""

    team: Optional[Team] = None
    build_phase: BuildPhase = BuildPhase.CARD_SELECTION
    current_slot_idx: int = 0
    last_score: float = 0.0
    final_approach_rate: Optional[int] = None