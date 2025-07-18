"""
This module defines the TrialState class, a data container for all dynamic
aspects of a single simulation trial.
"""

import uuid
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class TrialState:
    """
    Holds all the mutable state for a single simulation run.

    By centralizing the dynamic data, this class simplifies the main Trial
    orchestrator and makes the flow of state through the simulation explicit.
    """

    random_state: InitVar[np.random.Generator]

    # --- Core Gameplay State ---
    total_score: int = 0
    combo_count: int = 0
    perfect_hits: int = 0
    notes_hit: int = 0
    spawn_events_processed: int = 0
    song_end_time: float = field(default=0.0, init=False)

    # --- PPN & Stat Modifiers ---
    # This is initialized with the base PPN and updated by effects.
    current_slot_ppn: List[int] = field(default_factory=list)

    # --- Effect Trackers ---
    active_pl_count: int = 0
    pl_uptime_start_time: Optional[float] = None
    uptime_intervals: List[Tuple[float, float]] = field(default_factory=list)
    total_trick_end_time: float = 0.0

    active_sync_effects: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    active_appeal_boost: Optional[Dict[str, Any]] = None
    active_sru_effect: Optional[Dict[str, Any]] = None

    active_psu_effects: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=dict)
    active_cbu_effects: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=dict)
    active_spark_effects: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=dict)

    active_amp_boost: int = 0
    spark_charges: int = 0

    # --- Skill Activation State ---
    last_skill_info: Optional[Dict[str, Any]] = None
    score_skill_trackers: Dict[int, int] = field(default_factory=dict)
    year_group_skill_trackers: Dict[int, Set[str]] = field(default_factory=dict)

    # --- Miscellaneous State ---
    # Determines skill processing order for simultaneous activations.
    coin_flip: bool = field(init=False)
    hold_note_start_results: Dict[int, str] = field(default_factory=dict)
    song_has_ended: bool = False

    def __post_init__(self, random_state: np.random.Generator):
        """
        Initializes fields that require the random number generator.

        Args:
            random_state: The trial's seeded random number generator.
        """
        # The coin_flip determines skill processing order for an entire trial.
        self.coin_flip = random_state.random() < 0.5
