import warnings
from dataclasses import replace
from typing import Optional, Dict, Any, Union, List

from src.simulator.card.card_data import CardData
from src.simulator.card.gallery import Gallery
from src.simulator.card.stats import Stats
from src.simulator.core.skill import Skill
from src.simulator.core.leader_skill import LeaderSkill


class Card:
    """
    Represents a stateful, usable instance of a card. It is configured with
    a specific state (e.g., idolization, skill level) and provides dynamic
    properties based on that state.
    """

    def __init__(
        self,
        card_data: CardData,
        gallery: Gallery,
        level_cap_map: Dict,
        level_cap_bonus_map: Dict,
        idolized: bool = True,
        level: Optional[int] = None,
    ):
        self._data: CardData = card_data
        self._gallery: Gallery = gallery
        self._level_cap_map = level_cap_map
        self._level_cap_bonus_map = level_cap_bonus_map
        self.idolized_status: str = "idolized" if idolized else "unidolized"
        self._base_stats: Stats
        self.skill: Skill
        self.leader_skill: LeaderSkill

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
        self._base_stats = Stats(**stats_data)

        skill_data = self._data.skill
        self.skill = Skill(
            type=skill_data.get("type"),
            activation=skill_data.get("activation"),
            target=skill_data.get("target"),
            level=skill_data.get("level", []),
            thresholds=skill_data.get("threshold", []),
            chances=skill_data.get("chance", []),
            values=skill_data.get("value", []),
            durations=skill_data.get("duration", []),
        )

        leader_skill_data = self._data.leader_skill
        extra_data = leader_skill_data.get("extra", {})

        flat_leader_skill = {
            "attribute": leader_skill_data.get("leader_attribute"),
            "secondary_attribute": leader_skill_data.get("leader_secondary_attribute"),
            "value": leader_skill_data.get("leader_value", 0.0),
            "extra_attribute": extra_data.get("leader_extra_attribute"),
            "extra_target": extra_data.get("leader_extra_target"),
            "extra_value": extra_data.get("leader_extra_value", 0.0),
        }
        self.leader_skill = LeaderSkill(**flat_leader_skill)

    def _initialize_level(self, provided_level: Optional[int]) -> None:
        """Sets the card's level and applies any level-based stat bonuses."""
        self.level_cap: int = self._level_cap_map.get(self.rarity, {}).get(
            self.idolized_status, 1
        )
        self.level: int = self.level_cap

        if (
            self.rarity == "UR"
            and self.idolized_status == "idolized"
            and provided_level is not None
        ):
            if 100 <= provided_level <= 500:
                self.level = provided_level
            else:
                warnings.warn(
                    f"Custom level {provided_level} for UR is out of range (100-500). Using default level cap."
                )

        bonus_type = (
            "promo"
            if self.is_promo and not self.is_preidolized_non_promo
            else "non_promo"
        )
        self._apply_level_cap_bonus(bonus_type)

    def _apply_level_cap_bonus(self, bonus_type: str) -> None:
        """Adds level-based stat bonuses to the internal base stats."""
        bonus_map = self._level_cap_bonus_map.get(bonus_type, {})
        bonus_value = bonus_map.get(str(self.level), 0)
        if bonus_value > 0:
            self._base_stats = replace(
                self._base_stats,
                smile=self._base_stats.smile + bonus_value,
                pure=self._base_stats.pure + bonus_value,
                cool=self._base_stats.cool + bonus_value,
            )

    def _set_gallery_reference(self, gallery: Gallery) -> None:
        """
        Internal method to update the card's reference to the gallery object.
        Called by the parent Deck when its gallery is replaced.
        """
        self._gallery = gallery

    @property
    def stats(self) -> Stats:
        """
        Returns a new Stats object with the gallery bonus applied to the
        card's base stats. This will reflect the
        current state of the deck's gallery.
        """
        return Stats(
            smile=self._base_stats.smile + self._gallery.smile,
            pure=self._base_stats.pure + self._gallery.pure,
            cool=self._base_stats.cool + self._gallery.cool,
            sis_base=self._base_stats.sis_base,
            sis_max=self._base_stats.sis_max,
            image=self._base_stats.image,
        )

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
            raise ValueError(
                f"SIS slots must be between {self.stats.sis_base} and {self.stats.sis_max}."
            )
        self._current_sis_slots = value

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

    @property
    def skill_chance(self) -> Optional[float]:
        return self.get_skill_attribute_for_level(
            self.skill.chances, self.current_skill_level
        )

    @property
    def skill_value(self) -> Optional[Union[int, float]]:
        return self.get_skill_attribute_for_level(
            self.skill.values, self.current_skill_level
        )

    @property
    def skill_threshold(self) -> Optional[int]:
        return self.get_skill_attribute_for_level(
            self.skill.thresholds, self.current_skill_level
        )

    @property
    def skill_duration(self) -> Optional[Union[int, float]]:
        return self.get_skill_attribute_for_level(
            self.skill.durations, self.current_skill_level
        )

    def __repr__(self) -> str:
        """Provides a detailed, multi-line string representation of the card's state."""
        header = f"<Card id={self.card_id} name='{self.display_name}' rarity='{self.rarity}'>"

        info = (
            f"  - Info: Character='{self.character}', Attribute='{self.attribute}', "
            f"Level={self.level}, Idolized={self.idolized_status == 'idolized'}"
        )

        stats_line = (
            f"  - Stats (S/P/C): {self.stats.smile}/{self.stats.pure}/{self.stats.cool}"
        )

        # Skill Info
        skill_lines = [
            f"  - Skill: Level={self.current_skill_level}, Type='{self.skill.type}'"
        ]
        skill_details_parts = [
            f"Activation: '{self.skill.activation}'" if self.skill.activation else "",
            f"Target: '{self.skill.target}'" if self.skill.target else "",
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

        sis_line = f"  - SIS Slots: {self.current_sis_slots} (Base: {self.stats.sis_base}, Max: {self.stats.sis_max})"

        # Leader Skill Info
        ls = self.leader_skill
        ls_header = "  - Leader Skill:"
        ls_main_parts = [
            f"Boosts '{ls.attribute}'",
            f"based on '{ls.secondary_attribute}'" if ls.secondary_attribute else "",
            f"by {ls.value*100:.1f}%",
        ]
        ls_main = " ".join(part for part in ls_main_parts if part)

        ls_lines = [ls_header, f"    - Main: {ls_main}"]

        if ls.extra_attribute:
            ls_extra = (
                f"    - Extra: Boosts '{ls.extra_attribute}' for '{ls.extra_target}' "
                f"by {ls.extra_value*100:.1f}%"
            )
            ls_lines.append(ls_extra)

        return "\n".join([header, info, stats_line, *skill_lines, sis_line, *ls_lines])
