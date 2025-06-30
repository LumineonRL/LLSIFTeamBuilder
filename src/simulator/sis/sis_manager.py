import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

from src.simulator.sis.sis import SIS
from src.simulator.sis.sis_factory import SISFactory


@dataclass
class PlayerSIS:
    """Represents a unique SIS instance owned by a player."""

    manager_internal_id: int
    sis: SIS


class SISManager:
    """Manages a collection of player-owned School Idol Skills (SIS)."""

    def __init__(self, factory: SISFactory):
        self._factory = factory
        self._skills: Dict[int, PlayerSIS] = {}
        self._next_manager_internal_id: int = 1

    @property
    def skills(self):
        """Allows agent to directly read SIS skills."""
        return self._skills

    def add_sis(self, sid: int) -> Optional[int]:
        """Creates a SIS and adds it to the manager."""
        sis = self._factory.create_sis(sid)
        if not sis:
            return None

        manager_id = self._next_manager_internal_id
        self._skills[manager_id] = PlayerSIS(manager_id, sis)
        self._next_manager_internal_id += 1
        return manager_id

    def get_sis(self, manager_internal_id: int) -> Optional[SIS]:
        """Retrieves a SIS instance by its unique manager ID."""
        player_sis = self._skills.get(manager_internal_id)
        return player_sis.sis if player_sis else None

    def remove_sis(self, manager_internal_id: int) -> bool:
        """Removes an SIS by its unique manager ID."""
        if manager_internal_id in self._skills:
            del self._skills[manager_internal_id]
            return True
        else:
            warnings.warn(
                f"SIS Manager ID {manager_internal_id} does not exist and cannot be removed."
            )
            return False

    def get_unassigned_sis(self, assigned_sis_ids: Set[int]) -> List[PlayerSIS]:
        """Returns a list of PlayerSIS objects not present in the assigned set."""
        return [
            ps
            for manager_id, ps in self._skills.items()
            if manager_id not in assigned_sis_ids
        ]

    def get_player_sis(self, manager_internal_id: int) -> Optional[PlayerSIS]:
        """Retrieves a PlayerSIS wrapper instance by its unique manager ID."""
        return self._skills.get(manager_internal_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the manager's state to a dictionary."""
        return {
            "next_manager_internal_id": self._next_manager_internal_id,
            "skills": [
                {"manager_internal_id": ps.manager_internal_id, "sid": ps.sis.id}
                for ps in self._skills.values()
            ],
        }

    def save(self, filepath: str) -> bool:
        """Saves the current state to a JSON file."""
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except (IOError, TypeError) as e:
            warnings.warn(f"Error: Could not save SIS to {filepath}: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Loads the manager's state from a JSON file, overwriting the current state."""
        if not os.path.exists(filepath):
            warnings.warn(f"Load failed: File not found at {filepath}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            warnings.warn(f"Could not load or parse file {filepath}: {e}")
            return False

        self.delete()

        for item_data in state.get("skills", []):
            sis = self._factory.create_sis(sid=item_data["sid"])
            if sis:
                player_sis = PlayerSIS(item_data["manager_internal_id"], sis)
                self._skills[player_sis.manager_internal_id] = player_sis

        self._next_manager_internal_id = state.get("next_manager_internal_id", 1)
        return True

    def delete(self) -> None:
        """Clears all SIS from the manager."""
        self._skills.clear()
        self._next_manager_internal_id = 1

    def __repr__(self) -> str:
        """Provides a string summary of the SIS in the manager."""
        if not self._skills:
            return "<SISManager (empty)>"

        header = f"<SISManager ({len(self._skills)} skills)>"
        items = "\n".join(
            [
                f"  - ID {ps.manager_internal_id}: {ps.sis.name} (SID: {ps.sis.id})"
                for ps in self._skills.values()
            ]
        )
        return f"{header}\n{items}"
