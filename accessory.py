from typing import Optional, List, Union, Any

from accessorydata import AccessoryData
from accessorystats import AccessoryStats as Stats
from skill import Skill


class Accessory:
    """
    Represents a stateful instance of an accessory. It is configured with a
    single skill_level that determines both its stats and skill effects.
    The structure is aligned with the Card class for consistency.
    """

    def __init__(self, accessory_data: AccessoryData, skill_level: int = 1):
        self._data = accessory_data

        # --- Base Attributes ---
        self.accessory_id: int = self._data.accessory_id
        self.name: str = self._data.name
        self.character: str = self._data.character
        self.card_id: Optional[str] = self._data.card_id

        # --- Skill Initialization ---
        skill_data = self._data.skill
        trigger_data = skill_data.get("trigger", {})
        effect_data = skill_data.get("effect", {})

        self.skill = Skill(
            type=effect_data.get("type"),
            target=skill_data.get("target"),
            chances=trigger_data.get("chances", []),
            thresholds=trigger_data.get("values", []),
            durations=effect_data.get("durations", []),
            values=effect_data.get("values", []),
        )

        # --- Level and Stats Initialization ---
        self._skill_level: int = 1
        self.stats: Stats
        # The setter for skill_level will also initialize stats.
        self.skill_level = skill_level

    @property
    def skill_level(self) -> int:
        """
        The current level of the accessory (1-16). This single property
        controls both the stats and the skill effect values.
        """
        return self._skill_level

    @skill_level.setter
    def skill_level(self, value: int) -> None:
        """
        Sets the accessory's skill level and updates its stats accordingly.
        """
        value = int(value)
        if not 1 <= value <= 8:
            raise ValueError("Accessory skill_level must be between 1 and 8.")
        self._skill_level = value
        self._update_stats_from_skill_level()

    def _update_stats_from_skill_level(self) -> None:
        """Updates the accessory's stats based on its current skill_level."""
        index = self.skill_level - 1
        # Clamp index to protect against accessories with fewer than 16 stat entries
        clamped_index = max(0, min(index, len(self._data.stats) - 1))

        raw_stats = self._data.stats[clamped_index] if self._data.stats else [0, 0, 0]
        self.stats = Stats(smile=raw_stats[0], pure=raw_stats[1], cool=raw_stats[2])

    def get_skill_attribute_for_level(
        self, value_list: List[Any], level: int
    ) -> Optional[Any]:
        """
        Public method to get a skill attribute for a specific level.

        If the requested level is outside the bounds of the available data, it
        clamps to the nearest valid level (e.g., a level greater than the max
        will return the value for the max level, and a level < 1 will return
        the value for level 1).
        """
        if not value_list:
            return None

        index = level - 1
        clamped_index = max(0, min(index, len(value_list) - 1))

        return value_list[clamped_index]

    # --- Skill Properties (Aligned with Card properties) ---

    @property
    def skill_chance(self) -> Optional[float]:
        """Gets the skill's activation chance for the current skill level."""
        return self.get_skill_attribute_for_level(self.skill.chances, self.skill_level)

    @property
    def skill_value(self) -> Optional[Union[int, float]]:
        """Gets the skill's effect value for the current skill level."""
        return self.get_skill_attribute_for_level(self.skill.values, self.skill_level)

    @property
    def skill_threshold(self) -> Optional[int]:
        """Gets the skill's activation threshold for the current skill level."""
        return self.get_skill_attribute_for_level(
            self.skill.thresholds, self.skill_level
        )

    @property
    def skill_duration(self) -> Optional[Union[int, float]]:
        """Gets the skill's effect duration for the current skill level."""
        return self.get_skill_attribute_for_level(
            self.skill.durations, self.skill_level
        )

    def __repr__(self) -> str:
        """Provides a detailed string representation of the accessory's state."""
        header = (
            f"<Accessory id={self.accessory_id} name='{self.name}' "
            f"skill_level={self.skill_level}>"
        )
        stats_line = f"  - Stats: Smile={self.stats.smile}, Pure={self.stats.pure}, Cool={self.stats.cool}"

        # Skill Info
        skill_lines = [f"  - Skill: Type='{self.skill.type}'"]
        skill_details_parts = [
            f"Target: '{self.skill.target}'" if self.skill.target else ""
        ]
        skill_details = ", ".join(filter(None, skill_details_parts))
        if skill_details:
            skill_lines.append(f"    - Details: {skill_details}")

        skill_values_parts = [
            f"Chance: {self.skill_chance}%" if self.skill_chance is not None else "",
            (
                f"Threshold: {self.skill_threshold}"
                if self.skill_threshold is not None
                else ""
            ),
            f"Value: {self.skill_value}" if self.skill_value is not None else "",
            (
                f"Duration: {self.skill_duration}s"
                if self.skill_duration is not None
                else ""
            ),
        ]
        skill_values = ", ".join(filter(None, skill_values_parts))
        if skill_values:
            skill_lines.append(f"    - Effects: {skill_values}")

        return "\n".join([header, stats_line, *skill_lines])
