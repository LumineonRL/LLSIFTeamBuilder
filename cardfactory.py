import json
import numbers
from typing import Optional, Dict, Any, Union, List, Tuple
import warnings

from carddata import CardData
from card import Card
from gallery import Gallery


class CardFactory:
    def __init__(self, cards_json_path: str, level_caps_json_path: str, level_cap_bonuses_path: str):
        raw_card_data = self._load_json(cards_json_path)
        level_caps = self._load_json(level_caps_json_path)
        level_cap_bonuses = self._load_json(level_cap_bonuses_path)

        if not isinstance(raw_card_data, list):
            raise TypeError("Card data must be a list of objects.")
        if not isinstance(level_caps, dict):
            raise TypeError("Level caps data must be a dictionary.")
        if not isinstance(level_cap_bonuses, dict):
            raise TypeError("Level cap bonuses data must be a dictionary.")

        self._level_cap_map: Dict = level_caps
        self._level_cap_bonus_map: Dict = level_cap_bonuses
        self._card_data_map: Dict[int, CardData] = self._index_card_data(raw_card_data)

    def _load_json(self, json_path: str) -> Union[Dict, List]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load or parse JSON from {json_path}: {e}") from e

    def _index_card_data(self, raw_data: List[Dict]) -> Dict[int, CardData]:
        """Converts raw list of dicts into a map of CardData objects, keyed by card_id."""
        indexed_map = {}
        for record in raw_data:
            card_id = record.get('card_id')
            if not card_id or not isinstance(card_id, int):
                continue

            data_instance = CardData(
                card_id=int(card_id),
                display_name=str(record.get('display_name', 'Unknown')),
                rarity=str(record.get('rarity', 'N')),
                attribute=str(record.get('attribute', 'All')),
                character=str(record.get('character', 'Unknown')),
                is_promo=str(record.get("is_promo", "false")).lower() == 'true',
                is_preidolized_non_promo=str(record.get("is_preidolized_non_promo", "false")).lower() == 'true',
                stats=record.get('stats', {}),
                skill=record.get('skill', {}),
                leader_skill=record.get('leader_skill', {})
            )
            indexed_map[data_instance.card_id] = data_instance
        return indexed_map

    def _validate_and_sanitize_inputs(self, skill_level: Any, level: Any, sis_slots: Any) -> Tuple[int, Optional[int], Optional[int]]:
        """Validates input types, warns on failure, and returns sanitized values."""

        if not isinstance(skill_level, numbers.Integral):
            warnings.warn(f"Invalid type for skill_level: got {type(skill_level).__name__}, expected int. Defaulting to 1.")
            skill_level = 1

        if level is not None and not isinstance(level, numbers.Integral):
            warnings.warn(f"Invalid type for level: got {type(level).__name__}, expected int. Ignoring custom level.")
            level = None

        if sis_slots is not None and not isinstance(sis_slots, numbers.Integral):
            warnings.warn(f"Invalid type for sis_slots: got {type(sis_slots).__name__}, expected int. Ignoring custom SIS slots.")
            sis_slots = None

        return int(skill_level), int(level) if level is not None else None, int(sis_slots) if sis_slots is not None else None

    def _apply_skill_level(self, card: Card, skill_level: int) -> None:
        """Applies the skill level to the card, warning if out of range."""
        if 1 <= skill_level <= 8:
            card.current_skill_level = skill_level
        else:
            warnings.warn(f"Invalid skill level '{skill_level}'. Must be between 1 and 8. Defaulting to 1.")

    def _apply_sis_slots(self, card: Card, sis_slots: Optional[int]) -> None:
        """Applies SIS slots to the card, warning if out of range."""
        if sis_slots is not None:
            if card.stats.sis_base <= sis_slots <= card.stats.sis_max:
                card.current_sis_slots = sis_slots
            else:
                warnings.warn(
                    f"Invalid SIS slots value '{sis_slots}'. Must be between "
                    f"{card.stats.sis_base} and {card.stats.sis_max}. "
                    f"Defaulting to {card.current_sis_slots}."
                )

    def create_card(self, card_id: int, gallery: Gallery, idolized: bool = False,
                    skill_level: int = 1, level: Optional[int] = None,
                    sis_slots: Optional[int] = None) -> Optional[Card]:
        """Orchestrates the creation of a configured Card instance."""
        skill_level, level, sis_slots = self._validate_and_sanitize_inputs(skill_level, level, sis_slots)

        card_data = self._card_data_map.get(card_id)
        if not card_data:
            return None

        final_idolized = idolized
        if card_data.is_promo and not idolized:
            warnings.warn(f"Card ID {card_id} is a promo and cannot be unidolized. Forcing to idolized state.")
            final_idolized = True

        card = Card(card_data, gallery, self._level_cap_map, self._level_cap_bonus_map, final_idolized, level)

        self._apply_skill_level(card, skill_level)
        self._apply_sis_slots(card, sis_slots)

        return card
