"""
This module defines the TrialState class, a data container for all dynamic
aspects of a single simulation trial.
"""

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
    # This is a special initialization-only variable. It will be passed
    # to __post_init__ but will not become an instance field.
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
    # Manages the number of active Perfect Lock and Total Trick skills.
    active_pl_count: int = 0
    # Stores the start time when the first PL becomes active.
    pl_uptime_start_time: Optional[float] = None
    # A list of (start, end) tuples for each continuous PL uptime period.
    uptime_intervals: List[Tuple[float, float]] = field(default_factory=list)
    # The time at which the last Total Trick skill will expire.
    total_trick_end_time: float = 0.0

    # Stores active Sync effects, mapping a slot index to its effect data.
    active_sync_effects: Dict[int, Dict[str, Any]] = field(
        default_factory=dict)
    # Holds the currently active Appeal Boost effect data.
    active_appeal_boost: Optional[Dict[str, Any]] = None
    # Holds the currently active Skill Rate Up (SRU) effect data.
    active_sru_effect: Optional[Dict[str, Any]] = None
    # A list of all active Perfect Score Up (PSU) effects.
    active_psu_effects: List[Dict[str, Any]] = field(default_factory=list)
    # A list of all active Combo Bonus Up (CBU) effects.
    active_cbu_effects: List[Dict[str, Any]] = field(default_factory=list)

    # --- Amplify and Spark ---
    # Stores the pending skill level boost from Amplify skills.
    active_amp_boost: int = 0
    # Tracks charges for Spark accessory skills.
    spark_charges: int = 0
    # A list of all active Spark (tap score bonus) effects.
    active_spark_effects: List[Dict[str, Any]] = field(default_factory=list)

    # --- Skill Activation State ---
    # Stores info about the last skill activated, crucial for Encore.
    last_skill_info: Optional[Dict[str, Any]] = None
    # Tracks the next score threshold for each score-based skill.
    score_skill_trackers: Dict[int, int] = field(default_factory=dict)
    # Tracks remaining members needed for Year Group skills to activate.
    year_group_skill_trackers: Dict[int, Set[str]] = field(
        default_factory=dict)

    # --- Miscellaneous State ---
    # Determines skill processing order for simultaneous activations.
    coin_flip: bool = field(init=False)
    # Stores the judgement ('Perfect' or 'Great') for the start of hold notes.
    hold_note_start_results: Dict[int, str] = field(default_factory=dict)
    # Flag to stop the event loop once the song ends.
    song_has_ended: bool = False

    def __post_init__(self, random_state: np.random.Generator):
        """
        Initializes fields that require the random number generator.

        Args:
            random_state: The trial's seeded random number generator.
        """
        # The coin_flip determines skill processing order for an entire trial.
        self.coin_flip = random_state.random() < 0.5
