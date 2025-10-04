"""
This module defines abstract protocols for dependency inversion.

These protocols break circular import dependencies by allowing high-level
logic modules to depend on these abstractions rather than on concrete
implementation classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Protocol

if TYPE_CHECKING:
    import numpy as np
    from src.simulator.game_data import GameData
    from src.simulator.play_config import PlayConfig
    from src.simulator.sis import SIS
    from src.simulator.song.note import Note
    from src.simulator.song.song import Song
    from src.simulator.team.team import Team


class PlayInterface(Protocol):
    """
    An interface describing the properties and methods needed from Play.

    This protocol serves as a contract for simulation logic handlers,
    allowing them to access necessary context without creating a circular
    dependency on the concrete Play class.
    """

    # Properties
    team: Team
    song: Song
    config: PlayConfig
    game_data: GameData
    random_state: np.random.Generator
    trick_slots: Dict[int, List[SIS]]

    # Methods
    def get_note_multiplier(self, note: Note) -> float: ...

    def get_combo_multiplier(self, combo_count: int, game_data: GameData) -> float: ...
