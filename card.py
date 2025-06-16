import warnings
from dataclasses import replace
from typing import Optional, Dict, Any, Union, List

from carddata import CardData
from stats import Stats
from skill import Skill
from leaderskill import LeaderSkill

class Card:
    """
    Represents a stateful, usable instance of a card. It is configured with
    a specific state (e.g., idolization, skill level) and provides dynamic 
    properties based on that state.
    """
    def __init__(self, card_data: CardData, level_cap_map: Dict, level_cap_bonus_map: Dict, idolized: bool = True, level: Optional[int] = None):
        self._data: CardData = card_data
        self._level_cap_map = level_cap_map
        self._level_cap_bonus_map = level_cap_bonus_map
        self.idolized_status: str = "idolized" if idolized else "unidolized"
        self.stats: Stats
        self.skill: Skill

        self._initialize_base_attributes()
        self._initialize_nested_attributes()
        self._initialize_level(provided_level=level)

        self.current_skill_level = 1
        self.current_sis_slots = self.stats.sis_base

    def _initialize_base_attributes(self) -> None:
        """Copies basic, unchanging attributes from the CardData object."""
        self.card_id: int = self._data.card_id
        self.display_name: str = self._data.display_name
        self.rarity: str = self._data.rarity
        self.attribute: str = self._data.attribute
        self.character: str = self._data.character
        self.is_promo: bool = self._data.is_promo
        self.is_preidolized_non_promo: bool = self._data.is_preidolized_non_promo

    def _initialize_nested_attributes(self) -> None:
        """Initializes state-dependent nested objects like Stats, Skill, and LeaderSkill."""
        stats_data = self._data.stats.get(self.idolized_status, {})
        self.stats = Stats(**stats_data)

        skill_data = self._data.skill
        self.skill = Skill(
            type=skill_data.get('type'),
            activation=skill_data.get('activation'),
            target=skill_data.get('target'),
            level=skill_data.get('level', []),
            thresholds=skill_data.get('threshold', []),
            chances=skill_data.get('chance', []),
            values=skill_data.get('value', []),
            durations=skill_data.get('duration', [])
        )

        leader_skill_data = self._data.leader_skill
        flat_leader_skill = {
            "attribute": leader_skill_data.get("leader_attribute"),
            "secondary_attribute": leader_skill_data.get("leader_secondary_attribute"),
            "value": leader_skill_data.get("leader_value", 0.0)
        }
        self.leader_skill = LeaderSkill(**flat_leader_skill)

    def _initialize_level(self, provided_level: Optional[int]) -> None:
        """Sets the card's level and applies any level-based stat bonuses."""
        self.level_cap: int = self._level_cap_map.get(self.rarity, {}).get(self.idolized_status, 1)
        self.level: int = self.level_cap

        if self.rarity == 'UR' and self.idolized_status == 'idolized' and provided_level is not None:
            if 100 <= provided_level <= 500:
                self.level = provided_level
            else:
                warnings.warn(f"Custom level {provided_level} for UR is out of range (100-500). Using default level cap.")

        if self.is_promo and not self.is_preidolized_non_promo:
            self._update_stats_level_cap_promo()
        else:
            self._update_stats_level_cap_non_promo()

    def _update_stats_level_cap_non_promo(self) -> None:
        """Adds level-based stat bonuses for non-promo cards."""
        bonus_map = self._level_cap_bonus_map.get("non_promo", {})
        bonus_value = bonus_map.get(str(self.level), 0)
        if bonus_value > 0:
            self.stats = replace(self.stats,
                                 smile=self.stats.smile + bonus_value,
                                 pure=self.stats.pure + bonus_value,
                                 cool=self.stats.cool + bonus_value)

    def _update_stats_level_cap_promo(self) -> None:
        """Adds level-based stat bonuses for promo cards."""
        bonus_map = self._level_cap_bonus_map.get("promo", {})
        bonus_value = bonus_map.get(str(self.level), 0)
        if bonus_value > 0:
            self.stats = replace(self.stats,
                                 smile=self.stats.smile + bonus_value,
                                 pure=self.stats.pure + bonus_value,
                                 cool=self.stats.cool + bonus_value)

    @property
    def current_skill_level(self) -> int:
        return self._current_skill_level

    @current_skill_level.setter
    def current_skill_level(self, value: int) -> None:
        if not 1 <= value <= 8:
            raise ValueError("Skill level must be between 1 and 8.")
        self._current_skill_level = value

    @property
    def current_sis_slots(self) -> int:
        return self._current_sis_slots

    @current_sis_slots.setter
    def current_sis_slots(self, value: int) -> None:
        if not self.stats.sis_base <= value <= self.stats.sis_max:
            raise ValueError(f"SIS slots must be between {self.stats.sis_base} and {self.stats.sis_max}.")
        self._current_sis_slots = value

    def get_skill_attribute_for_level(self, value_list: List[Any], level: int) -> Optional[Any]:
        """Public method to get a skill attribute for a specific level (1-16)."""
        if not 1 <= level <= 16:
            warnings.warn(f"Requested skill level {level} is outside the valid data range (1-16).")
            return None
        index = level - 1
        return value_list[index] if 0 <= index < len(value_list) else None

    @property
    def skill_chance(self) -> Optional[float]:
        return self.get_skill_attribute_for_level(self.skill.chances, self.current_skill_level)

    @property
    def skill_value(self) -> Optional[Union[int, float]]:
        return self.get_skill_attribute_for_level(self.skill.values, self.current_skill_level)

    @property
    def skill_threshold(self) -> Optional[int]:
        return self.get_skill_attribute_for_level(self.skill.thresholds, self.current_skill_level)

    @property
    def skill_duration(self) -> Optional[Union[int, float]]:
        return self.get_skill_attribute_for_level(self.skill.durations, self.current_skill_level)

    def __repr__(self) -> str:
        return f"<Card id={self.card_id} name='{self.display_name}' rarity='{self.rarity}'>"
