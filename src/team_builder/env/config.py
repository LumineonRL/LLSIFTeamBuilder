import json
from pathlib import Path
from typing import Dict, List, Any


def _load_json(path: Path) -> Any:
    """Loads a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load required data file: {e}") from e


def _create_map_from_list(items: List[str]) -> Dict[str, int]:
    """Creates a dictionary mapping items to their indices."""
    return {name: i for i, name in enumerate(sorted(items))}


def _get_skill_related_map(
    card_data: List[Dict], key: str, sub_key: str
) -> Dict[str, int]:
    """
    Creates a mapping for skill-related attributes from card data.
    """
    items = set(
        c[key][sub_key] for c in card_data if c.get(key) and c[key].get(sub_key)
    )
    return _create_map_from_list(list(items))


class EnvConfig:
    """
    Configuration class for the environment, holding constants and mappings.
    """

    MAX_CARDS_IN_DECK = 200
    MAX_SKILL_LIST_ENTRIES = 16

    MAX_STAT_VALUE = 250000.0
    MAX_LS_VALUE = 0.15
    MAX_SKILL_THRESHOLD = 18000.0
    MAX_SKILL_VALUE = 150000.0
    MAX_SKILL_DURATION = 50.0

    RARITY_MAP = {"N": 0, "R": 1, "SR": 2, "SSR": 3, "UR": 4}
    ATTRIBUTE_MAP = {"Smile": 0, "Pure": 1, "Cool": 2}
    LEADER_ATTRIBUTE_MAP = {"Smile": 0, "Pure": 1, "Cool": 2, "None": 3}

    def __init__(self, data_path: str = "data"):
        """
        Initializes the configuration by loading data and creating mappings.

        Args:
            data_path: Path to the directory containing required data files.
        """
        base_path = Path(data_path)
        card_data = _load_json(base_path / "cards.json")
        ls_target_data = _load_json(base_path / "additional_leader_skill_map.json")

        # --- Dynamic Categorical Mappings ---
        self.character_map = self._create_character_map(card_data)
        self.character_other_index = len(self.character_map)
        self.ls_extra_target_map = _create_map_from_list(list(ls_target_data.keys()))
        self.skill_type_map = _get_skill_related_map(card_data, "skill", "type")
        self.skill_activation_map = _get_skill_related_map(
            card_data, "skill", "activation"
        )

    @staticmethod
    def _create_character_map(card_data: List[Dict]) -> Dict[str, int]:
        """Returns a list of unique character names among all non-N cards."""
        characters = set(c["character"] for c in card_data if c["rarity"] != "N")
        return _create_map_from_list(list(characters))
