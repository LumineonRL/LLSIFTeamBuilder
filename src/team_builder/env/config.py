import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import numpy.typing as npt


def _load_json(path: Path) -> Any:
    """Loads a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load required data file: {e}") from e


def _get_skill_related_map(
    card_data: List[Dict], key: str, sub_key: str
) -> Dict[str, int]:
    """
    Creates a mapping for skill-related attributes from card data.
    """
    items = set(
        c[key][sub_key] for c in card_data if c.get(key) and c[key].get(sub_key)
    )
    return {name: i for i, name in enumerate(sorted(items))}


class EnvConfig:
    """
    Configuration class for the environment, holding constants and mappings.
    """

    MAX_CARDS_IN_DECK = 500
    MAX_SKILL_LIST_ENTRIES = 16
    MAX_SKILL_LEVEL = 8
    MAX_SIS_SLOTS = 8

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
        ls_map_data = _load_json(base_path / "additional_leader_skill_map.json")

        main_characters = ls_map_data.get("All", [])
        if not main_characters:
            raise ValueError(
                "'All' key not found or empty in additional_leader_skill_map.json"
            )

        self.character_map = self._create_character_map(main_characters)
        self.character_other_index = 0

        self.ls_extra_target_map: Dict[str, npt.NDArray[np.float32]] = (
            self._create_multi_hot_target_map(
                ls_map_data, self.character_map, self.character_other_index
            )
        )
        self.ls_extra_target_default_vector: npt.NDArray[np.float32] = np.zeros(
            len(self.character_map) + 1, dtype=np.float32
        )
        self.skill_type_map = _get_skill_related_map(card_data, "skill", "type")
        self.skill_activation_map = _get_skill_related_map(
            card_data, "skill", "activation"
        )

    @staticmethod
    def _create_character_map(main_characters: List[str]) -> Dict[str, int]:
        """
        Creates a mapping for main characters, reserving index 0 for 'Other'.

        "Main characters" are characters from u's, Aquours, Niji, and Liella.
        Note that this currently treats characters such as the N-version of
        the Niji girls or April Fools versions of u's and Aqours as the
        same character when it shouldn't.
        """
        return {name: i + 1 for i, name in enumerate(sorted(main_characters))}

    @staticmethod
    def _create_multi_hot_target_map(
        ls_map_data: Dict[str, List[str]],
        character_map: Dict[str, int],
        character_other_index: int,
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Creates a map from a group name (e.g., 'first-year') to a pre-computed
        multi-hot vector representing the characters in that group.
        The vector's order and length are determined by the main character_map.
        """
        multi_hot_map: Dict[str, npt.NDArray[np.float32]] = {}
        num_characters = len(character_map) + 1  # +1 for 'Other'

        for group_name, character_list in ls_map_data.items():
            if group_name == "All":
                continue

            group_vector: npt.NDArray[np.float32] = np.zeros(
                num_characters, dtype=np.float32
            )
            for character_name in character_list:
                char_index = character_map.get(character_name, character_other_index)

                if char_index != character_other_index:
                    group_vector[char_index] = 1.0

            multi_hot_map[group_name] = group_vector

        return multi_hot_map
