import json
import warnings
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any

from src.simulator.core.leader_skill import LeaderSkill


@dataclass(frozen=True)
class GuestData:
    """
    Represents the static, immutable data for a single guest leader skill,
    loaded from the unique_leader_skills.json file.
    """

    leader_skill_id: int
    leader_attribute: Optional[str] = None
    leader_secondary_attribute: Optional[str] = None
    leader_value: Optional[float] = None
    leader_extra_attribute: Optional[str] = None
    leader_extra_target: Optional[str] = None
    leader_extra_value: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuestData":
        """Creates a GuestData instance from a dictionary, flattening the 'extra' key."""
        extra_data = data.get("extra", {})
        return cls(
            leader_skill_id=data["leader_skill_id"],
            leader_attribute=data.get("leader_attribute"),
            leader_secondary_attribute=data.get("leader_secondary_attribute"),
            leader_value=data.get("leader_value"),
            leader_extra_attribute=extra_data.get("leader_extra_attribute"),
            leader_extra_target=extra_data.get("leader_extra_target"),
            leader_extra_value=extra_data.get("leader_extra_value"),
        )


class Guest:
    """Manages the active guest leader skill for a team."""

    def __init__(self, unique_skills_path: str):
        self._all_guests: Dict[int, GuestData] = self._load_and_index_guests(
            unique_skills_path
        )
        self.current_guest: Optional[GuestData] = None

    def _load_and_index_guests(self, filepath: str) -> Dict[int, GuestData]:
        """Loads the JSON file and indexes the guest data by ID."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise TypeError("Guest data must be a list of objects.")

            return {item["leader_skill_id"]: GuestData.from_dict(item) for item in data}
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(f"Error loading or parsing guest data from {filepath}: {e}")
            return {}

    def set_guest(self, leader_skill_id: int) -> bool:
        """
        Sets the active guest by looking up their leader_skill_id.

        Args:
            leader_skill_id: The ID of the guest leader skill to set.

        Returns:
            True if the guest was found and set, False otherwise.
        """
        guest_data = self._all_guests.get(leader_skill_id)
        if guest_data:
            self.current_guest = guest_data
            return True
        self.current_guest = None
        warnings.warn(f"Guest with ID {leader_skill_id} not found.")
        return False

    @property
    def leader_skill(self) -> Optional[LeaderSkill]:
        """Returns a LeaderSkill object for the current guest, if any."""
        if not self.current_guest:
            return None

        return LeaderSkill(
            attribute=self.current_guest.leader_attribute,
            secondary_attribute=self.current_guest.leader_secondary_attribute,
            value=self.current_guest.leader_value or 0.0,
            extra_attribute=self.current_guest.leader_extra_attribute,
            extra_target=self.current_guest.leader_extra_target,
            extra_value=self.current_guest.leader_extra_value or 0.0,
        )

    def __repr__(self) -> str:
        if not self.current_guest:
            return "<Guest active_id=None>"

        header = "--- Current Guest Details ---"
        footer = "---------------------------"

        details = []
        for field in fields(self.current_guest):
            value = getattr(self.current_guest, field.name)
            display_name = field.name.replace("_", " ").title()
            display_value = str(value) if value is not None else "N/A"
            details.append(f"{display_name}: {display_value}")

        return f"{header}\n" + "\n".join(details) + f"\n{footer}"
