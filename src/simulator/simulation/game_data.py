"""
This module defines the GameData class, responsible for loading and holding
all static game data from JSON files. This object is intended to be created
once and shared across simulations.
"""

import json
import os
import warnings
from typing import Dict, List, Set, Tuple


class GameData:
    """
    Loads and holds all static game data from JSON files.

    The 'too-few-public-methods' warning is disabled as this class is
    intentionally designed as a data container.

    Attributes:
        HEAL_MULTIPLIER (int): The multiplier used to convert heal values to
                               score.
        MAX_COMBO_FEVER_BONUS (int): The maximum score bonus achievable from
                                     a Combo Fever skill effect per note.
        group_mapping (Dict[str, Set[str]]): Maps leader skill target groups
                                             to character names.
        sub_group_mapping (Dict[str, Set[str]]): Maps skill target subgroups
                                                 (like year or unit) to
                                                 character names.
        year_group_mapping (Dict[str, Set[str]]): Maps year groups to
                                                  character names.
        combo_bonus_tiers (List[Tuple[int, float]]): A sorted list of combo
                                                     thresholds and their
                                                     corresponding score
                                                     multipliers.
        note_speed_map (Dict[int, float]): Maps approach rate (1-10) to the
                                           on-screen duration of a note.
        combo_fever_map (List[Tuple[int, float]]): A sorted list of combo
                                                   thresholds for the Combo
                                                   Fever skill and their
                                                   score multipliers.
    """

    HEAL_MULTIPLIER = 480
    MAX_COMBO_FEVER_BONUS = 1000

    def __init__(self, data_path: str = "data"):
        """
        Initializes GameData by loading all necessary data files.

        Args:
            data_path (str): The path to the directory containing data files.
        """
        self.group_mapping = self._load_json_mapping(
            os.path.join(data_path, "additional_leader_skill_map.json"),
            "additional leader skill map for PPN calculation",
        )
        self.sub_group_mapping = self._load_json_mapping(
            os.path.join(data_path, "year_group_target.json"),
            "year group mapping for skills",
        )
        # self.year_group_mapping = self._load_json_mapping(
        #     os.path.join(data_path, "year_group_mapping.json"),
        #     "year group mapping for skills",
        # )
        self.combo_bonus_tiers = self._load_combo_bonuses(
            os.path.join(data_path, "combo_bonuses.json")
        )
        self.note_speed_map = self._load_note_speed_map(
            os.path.join(data_path, "note_speed_map.json")
        )
        self.combo_fever_map = self._load_combo_fever_map(
            os.path.join(data_path, "combo_fever_map.json")
        )

    def _load_json_mapping(self, filepath: str, name: str) -> Dict[str, Set[str]]:
        """
        Generic helper to load a JSON file mapping groups to character sets.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            return {
                group.strip(): set(characters)
                for group, characters in raw_mapping.items()
            }
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            warnings.warn(f"Could not load {name} from '{filepath}': {e}.")
            return {}

    def _load_combo_bonuses(self, filepath: str) -> List[Tuple[int, float]]:
        """Loads and sorts combo bonus tiers from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            # Sort descending by combo count to simplify lookup
            return sorted([(int(k), v) for k, v in raw_mapping.items()], reverse=True)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load combo bonuses from '{filepath}': {e}.")
            return [(0, 1.0)]

    def _load_note_speed_map(self, filepath: str) -> Dict[int, float]:
        """Loads the note speed to on-screen duration mapping."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            return {int(k): v for k, v in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load note speed map from '{filepath}': {e}.")
            # Provide a default mapping if file is missing/corrupt
            return {
                i: (1.6 - i * 0.1) if i >= 6 else (1.9 - i * 0.15) for i in range(1, 11)
            }

    def _load_combo_fever_map(self, filepath: str) -> List[Tuple[int, float]]:
        """Loads and sorts the combo fever multiplier map from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            # Sort descending by combo count to simplify lookup
            return sorted([(int(k), v) for k, v in raw_mapping.items()], reverse=True)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load combo fever map from '{filepath}': {e}.")
            return []
