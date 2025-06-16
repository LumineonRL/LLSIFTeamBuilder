import warnings
from typing import Optional, List, Union

from accessorydata import AccessoryData
from accessorystats import AccessoryStats as Stats

class Accessory:
    """
    Represents a stateful instance of an accessory, configured with a specific level.
    """
    def __init__(self, accessory_data: AccessoryData, level: int = 1):
        self._data = accessory_data

        self.accessory_id: int = self._data.accessory_id
        self.name: str = self._data.name
        self.character: str = self._data.character
        self.card_id: Optional[str] = self._data.card_id
        self.skill_type: str = self._data.skill.get("effect", {}).get("type", "Unknown")

        self._skill_trigger = self._data.skill.get("trigger", {})
        self._skill_effect = self._data.skill.get("effect", {})
        self.skill_trigger_chances: List[float] = self._skill_trigger.get("chances", [])
        self.skill_trigger_values: List[float] = self._skill_trigger.get("values", [])
        self.skill_effect_durations: List[float] = self._skill_effect.get("durations", [])
        self.skill_effect_values: List[float] = self._skill_effect.get("values", [])

        self._level: int = 1
        self.stats: Stats
        self.level = level
        self.level = level

    @property
    def level(self) -> int:
        """The current level of the accessory (1-8)."""
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        if not 1 <= value <= 8:
            raise ValueError("Accessory level must be between 1 and 8.")
        self._level = value
        self._update_stats_from_level()

    def _update_stats_from_level(self) -> None:
        """Updates the accessory's stats based on its current level."""
        index = self.level - 1
        raw_stats = self._data.stats[index] if 0 <= index < len(self._data.stats) else [0, 0, 0]
        self.stats = Stats(smile=raw_stats[0], pure=raw_stats[1], cool=raw_stats[2])

    def get_skill_value_for_level(self, value_list: List[Union[int, float]], level: int) -> Union[int, float]:
        """
        Public method to get a skill value from a list for a specific level.
        Returns 0.0 if the level is out of bounds.
        """
        index = level - 1
        if not 1 <= level <= 16: # Accessories can have data up to level 16
            warnings.warn(f"Requested level {level} is outside the valid range (1-16).")
            return 0.0
        return value_list[index] if 0 <= index < len(value_list) else 0.0

    @property
    def current_skill_trigger_chance(self) -> float:
        return self.get_skill_value_for_level(self.skill_trigger_chances, self.level)

    @property
    def current_skill_trigger_value(self) -> float:
        return self.get_skill_value_for_level(self.skill_trigger_values, self.level)

    @property
    def current_skill_effect_duration(self) -> float:
        return self.get_skill_value_for_level(self.skill_effect_durations, self.level)

    @property
    def current_skill_effect_value(self) -> float:
        return self.get_skill_value_for_level(self.skill_effect_values, self.level)

    def __repr__(self) -> str:
        header = f"<Accessory id={self.accessory_id} card_id={self.card_id} name='{self.name}' level={self.level}>"
        stats_line = f"  - Stats: Smile={self.stats.smile}, Pure={self.stats.pure}, Cool={self.stats.cool}"
        skill_line = f"  - Skill Type: {self.skill_type}"
        trigger_line = f"  - Trigger: {self.current_skill_trigger_chance}% chance, Value: {self.current_skill_trigger_value}"
        effect_line = f"  - Effect Value: {self.current_skill_effect_value}, Duration: {self.current_skill_effect_duration}s"
        return f"{header}\n{stats_line}\n{skill_line}\n{trigger_line}\n{effect_line}"
