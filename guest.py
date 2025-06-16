import json
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    def from_dict(cls, data: Dict[str, Any]) -> 'GuestData':
        """Creates a GuestData instance from a dictionary, flattening the 'extra' key."""
        extra_data = data.get('extra', {})
        return cls(
            leader_skill_id=data['leader_skill_id'],
            leader_attribute=data.get('leader_attribute'),
            leader_secondary_attribute=data.get('leader_secondary_attribute'),
            leader_value=data.get('leader_value'),
            leader_extra_attribute=extra_data.get('leader_extra_attribute'),
            leader_extra_target=extra_data.get('leader_extra_target'),
            leader_extra_value=extra_data.get('leader_extra_value')
        )

class Guest:
    """Manages the active guest leader skill for a team."""

    def __init__(self, unique_skills_path: str):
        self._all_guests: Dict[int, GuestData] = self._load_and_index_guests(unique_skills_path)
        self.current_guest: Optional[GuestData] = None

    def _load_and_index_guests(self, filepath: str) -> Dict[int, GuestData]:
        """Loads the JSON file and indexes the guest data by ID."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise TypeError("Guest data must be a list of objects.")

            return {item['leader_skill_id']: GuestData.from_dict(item) for item in data}
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
        else:
            self.current_guest = None
            warnings.warn(f"Guest with ID {leader_skill_id} not found.")
            return False

    def display_guest(self) -> None:
        """Prints the details of the currently active guest."""
        if not self.current_guest:
            print("No active guest is set.")
            return

        def _format_value(value: Any) -> str:
            """Helper function to format None values gracefully for display."""
            return str(value) if value is not None else "N/A"

        print("--- Current Guest Details ---")
        for attr, value in self.current_guest.__dict__.items():
            display_name = attr.replace('_', ' ').title()
            print(f"{display_name}: {_format_value(value)}")
        print("---------------------------")

    def __repr__(self) -> str:
        if self.current_guest:
            return f"<Guest active_id={self.current_guest.leader_skill_id}>"
        return "<Guest active_id=None>"
