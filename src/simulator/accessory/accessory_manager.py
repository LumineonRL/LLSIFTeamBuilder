import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

from src.simulator.accessory.accessory import Accessory
from src.simulator.accessory.accessory_factory import AccessoryFactory


@dataclass
class PlayerAccessory:
    """Represents a unique accessory instance owned by a player."""

    manager_internal_id: int
    accessory: Accessory


class AccessoryManager:
    """Manages a collection of player-owned accessories."""

    def __init__(self, factory: AccessoryFactory):
        self._factory = factory
        self._accessories: Dict[int, PlayerAccessory] = {}
        self._next_manager_internal_id: int = 1

    def add_accessory(self, accessory_id: int, skill_level: int = 1) -> Optional[int]:
        """Creates an accessory and adds it to the manager."""
        # Pass skill_level to the factory
        accessory = self._factory.create_accessory(
            accessory_id, skill_level=skill_level
        )
        if not accessory:
            return None

        manager_id = self._next_manager_internal_id
        self._accessories[manager_id] = PlayerAccessory(manager_id, accessory)
        self._next_manager_internal_id += 1
        return manager_id

    def get_accessory(self, manager_internal_id: int) -> Optional[Accessory]:
        """Retrieves an accessory instance by its unique manager ID."""
        player_accessory = self._accessories.get(manager_internal_id)
        return player_accessory.accessory if player_accessory else None

    def remove_accessory(self, manager_internal_id: int) -> bool:
        """Removes an accessory by its unique manager ID."""
        if manager_internal_id in self._accessories:
            del self._accessories[manager_internal_id]
            return True
        else:
            warnings.warn(
                f"Accessory Manager ID {manager_internal_id} does not exist and cannot be removed."
            )
            return False

    def modify_accessory(
        self, manager_internal_id: int, skill_level: Optional[int] = None
    ) -> bool:
        """Modifies the state of an accessory in the manager."""
        player_accessory = self._accessories.get(manager_internal_id)
        if not player_accessory:
            warnings.warn(f"Accessory with Manager ID {manager_internal_id} not found.")
            return False

        try:
            if skill_level is not None:
                player_accessory.accessory.skill_level = skill_level
        except ValueError as e:
            warnings.warn(f"Error modifying accessory {manager_internal_id}: {e}")
            return False
        return True

    def get_unassigned_accessories(
        self, assigned_accessory_ids: Set[int]
    ) -> List[PlayerAccessory]:
        """Returns a list of PlayerAccessory objects not in the assigned set."""
        return [
            acc
            for manager_id, acc in self._accessories.items()
            if manager_id not in assigned_accessory_ids
        ]

    def get_player_accessory(
        self, manager_internal_id: int
    ) -> Optional[PlayerAccessory]:
        """Retrieves a PlayerAccessory wrapper instance by its unique manager ID."""
        return self._accessories.get(manager_internal_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the manager's state to a dictionary."""
        return {
            "next_manager_internal_id": self._next_manager_internal_id,
            "accessories": [
                {
                    "manager_internal_id": pa.manager_internal_id,
                    "accessory_id": pa.accessory.accessory_id,
                    "skill_level": pa.accessory.skill_level,  # Save only skill_level
                }
                for pa in self._accessories.values()
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
            warnings.warn(f"Error: Could not save accessories to {filepath}: {e}")
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

        for item_data in state.get("accessories", []):
            # Load only skill_level
            accessory = self._factory.create_accessory(
                accessory_id=item_data["accessory_id"],
                skill_level=item_data.get("skill_level", 1),
            )
            if accessory:
                player_acc = PlayerAccessory(
                    item_data["manager_internal_id"], accessory
                )
                self._accessories[player_acc.manager_internal_id] = player_acc

        self._next_manager_internal_id = state.get("next_manager_internal_id", 1)
        return True

    def delete(self) -> None:
        """Clears all accessories from the manager."""
        self._accessories.clear()
        self._next_manager_internal_id = 1

    def __repr__(self) -> str:
        """Provides a string summary of the accessories in the manager."""
        if not self._accessories:
            return "<AccessoryManager (empty)>"

        header = f"<AccessoryManager ({len(self._accessories)} accessories)>"
        items = "\n".join(
            [
                f"  - ID {pa.manager_internal_id}: {pa.accessory}"
                for pa in self._accessories.values()
            ]
        )
        return f"{header}\n{items}"
