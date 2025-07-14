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

    MAX_CARDS_IN_DECK = 100
    MAX_ACCESSORIES_IN_INVENTORY = 100
    MAX_SIS_IN_INVENTORY = 150
    MAX_GUESTS = 313
    MAX_SKILL_LIST_ENTRIES = 16
    MAX_SKILL_LEVEL = 8
    MAX_SIS_SLOTS = 8
    MAX_EQUIPPABLE_SIS = 5

    MAX_STAT_VALUE = 250000.0
    MAX_TEAM_STAT_VALUE = 500000.0
    MAX_ACCESSORY_STAT = 8500.0
    MAX_LS_VALUE = 0.15
    MAX_SKILL_DURATION = 50.0

    MAX_SKILL_VALUE_DEFAULT = 150000.0  # For "Scorer"
    MAX_SKILL_VALUE_PERCENT = 4.0  # For "Appeal Boost", "Skill Rate Up"
    MAX_SKILL_VALUE_AMP = 16.0  # For "Amplify"
    MAX_SKILL_VALUE_HEAL = 70.0  # For "Healer"
    MAX_SKILL_VALUE_FLAT = 12000.0  # For "Combo Bonus Up", "Perfect Score Up", "Spark"

    MAX_SKILL_THRESHOLD_DEFAULT = 92.0  # For most types (Notes, Combo, Perfect)
    MAX_SKILL_THRESHOLD_SCORE = 20000.0  # For "Score"
    MAX_SKILL_THRESHOLD_TIME = 30.0  # For "Time"
    MAX_ACC_SKILL_THRESHOLD = 10.0  # For "Spark"

    MAX_SIS_VALUE_ALL_PERCENT = 0.042
    MAX_SIS_VALUE_CHARM = 1.5
    MAX_SIS_VALUE_HEAL = 480.0
    MAX_SIS_VALUE_SELF_FLAT = 1400.0
    MAX_SIS_VALUE_SELF_PERCENT = 0.53
    MAX_SIS_VALUE_TRICK = 0.33

    MAX_SONG_LENGTH = 175.0
    MAX_NOTE_COUNT = 1061

    RARITY_MAP = {"N": 0, "R": 1, "SR": 2, "SSR": 3, "UR": 4}
    ATTRIBUTE_MAP = {"Smile": 0, "Pure": 1, "Cool": 2}

    SIS_EFFECT_MAP = {
        "all percent boost": 0,
        "charm": 1,
        "heal": 2,
        "self flat boost": 3,
        "self percent boost": 4,
        "trick": 5,
    }

    ACCESSORY_SKILL_TYPE_MAP = {
        "Appeal Boost": 0,
        "Healer": 1,
        "Perfect Lock": 2,
        "Sync": 3,
        "Perfect Score Up": 4,
        "Skill Rate Up": 5,
        "Combo Bonus Up": 6,
        "Encore": 7,
        "Amplify": 8,
        "Scorer": 9,
        "Spark": 10,
    }

    def __init__(self, data_path: str = "data"):
        """
        Initializes the configuration by loading data and creating mappings.


        Args:
            data_path: Path to the directory containing required data files.
        """
        base_path = Path(data_path)
        card_data = _load_json(base_path / "cards.json")
        ls_map_data = _load_json(base_path / "additional_leader_skill_map.json")
        skill_target_map_data = _load_json(base_path / "year_group_target.json")

        main_characters = ls_map_data.get("All", [])
        if not main_characters:
            raise ValueError(
                "'All' key not found or empty in additional_leader_skill_map.json"
            )

        self.character_map = self._create_character_map(main_characters)

        self.ls_extra_target_map: Dict[str, npt.NDArray[np.float32]] = (
            self._create_multi_hot_target_map(ls_map_data, self.character_map)
        )
        self.ls_extra_target_default_vector: npt.NDArray[np.float32] = np.zeros(
            len(self.character_map), dtype=np.float32
        )

        self.skill_target_map: Dict[str, npt.NDArray[np.float32]] = (
            self._create_multi_hot_target_map(skill_target_map_data, self.character_map)
        )
        self.skill_target_default_vector: npt.NDArray[np.float32] = np.zeros(
            len(self.character_map), dtype=np.float32
        )

        self.sis_equip_restriction_map: Dict[str, npt.NDArray[np.float32]] = (
            self._create_sis_equip_restriction_map(ls_map_data, self.character_map)
        )
        self.sis_equip_restriction_default_vector: npt.NDArray[np.float32] = np.zeros(
            len(self.character_map), dtype=np.float32
        )

        self.sis_group_map: Dict[str, npt.NDArray[np.float32]] = (
            self._create_multi_hot_target_map(ls_map_data, self.character_map)
        )
        self.sis_group_default_vector: npt.NDArray[np.float32] = np.zeros(
            len(self.character_map), dtype=np.float32
        )

        self.skill_type_map = _get_skill_related_map(card_data, "skill", "type")
        self.skill_activation_map = _get_skill_related_map(
            card_data, "skill", "activation"
        )

    @staticmethod
    def _create_character_map(main_characters: List[str]) -> Dict[str, int]:
        """
        Creates a mapping for main characters. An all-zero vector will represent 'Other'.


        "Main characters" are characters from u's, Aquours, Niji, and Liella.
        Note that this currently treats characters such as the N-version of
        the Niji girls or April Fools versions of u's and Aqours as the
        same character when it shouldn't.
        """
        return {name: i for i, name in enumerate(sorted(main_characters))}

    @staticmethod
    def _create_multi_hot_target_map(
        map_data: Dict[str, List[str]],
        character_map: Dict[str, int],
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Creates a map from a group name (e.g., 'first-year') to a pre-computed
        multi-hot vector representing the characters in that group.
        The vector's order and length are determined by the main character_map.
        """
        multi_hot_map: Dict[str, npt.NDArray[np.float32]] = {}
        num_characters = len(character_map)

        for group_name, character_list in map_data.items():
            if group_name == "All":
                continue

            group_vector: npt.NDArray[np.float32] = np.zeros(
                num_characters, dtype=np.float32
            )
            for character_name in character_list:
                char_index = character_map.get(character_name)
                if char_index is not None:
                    group_vector[char_index] = 1.0

            multi_hot_map[group_name] = group_vector

        return multi_hot_map

    @staticmethod
    def _create_sis_equip_restriction_map(
        map_data: Dict[str, List[str]],
        character_map: Dict[str, int],
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Creates a map from an SIS equip restriction group to a pre-computed
        multi-hot vector. This is for characters and years.
        The vector size is the number of characters.
        """
        num_characters = len(character_map)
        multi_hot_map: Dict[str, npt.NDArray[np.float32]] = {}

        for group_name, character_list in map_data.items():
            if group_name == "All":
                continue

            if group_name in ["Smile", "Pure", "Cool"]:
                continue

            group_vector = np.zeros(num_characters, dtype=np.float32)
            for character_name in character_list:
                char_index = character_map.get(character_name)
                if char_index is not None:
                    group_vector[char_index] = 1.0
            multi_hot_map[group_name] = group_vector

        return multi_hot_map
