import json
import warnings
from typing import Dict, Optional, Any

from src.simulator.accessory.accessory_data import AccessoryData
from src.simulator.accessory.accessory import Accessory


class AccessoryFactory:
    """
    Handles loading accessory data from JSON and creating Accessory instances.
    """

    def __init__(self, accessories_json_path: str):
        raw_data = self._load_json(accessories_json_path)
        if not isinstance(raw_data, dict):
            raise TypeError("Accessories data file must be a dictionary of objects.")
        self._accessory_data_map: Dict[int, AccessoryData] = self._index_accessory_data(
            raw_data
        )

    def _load_json(self, json_path: str) -> Dict:
        """Helper to load and parse a JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to load or parse JSON from {json_path}: {e}"
            ) from e

    def _index_accessory_data(
        self, raw_data: Dict[str, Any]
    ) -> Dict[int, AccessoryData]:
        """Converts the raw dictionary data into AccessoryData objects."""
        indexed_map = {}
        for acc_id_str, record in raw_data.items():
            try:
                acc_id = int(acc_id_str)
                data_instance = AccessoryData(
                    accessory_id=acc_id,
                    name=record.get("name", "Unknown Accessory"),
                    character=record.get("character", "Unknown"),
                    card_id=record.get("card_id"),
                    stats=record.get("stats", []),
                    skill=record.get("skill", {}),
                )
                indexed_map[acc_id] = data_instance
            except (ValueError, TypeError) as e:
                warnings.warn(
                    f"Warning: Skipping invalid accessory record with key '{acc_id_str}': {e}"
                )
        return indexed_map

    def create_accessory(
        self, accessory_id: int, skill_level: int = 1
    ) -> Optional[Accessory]:
        """Creates a configured, stateful Accessory instance."""
        accessory_data = self._accessory_data_map.get(accessory_id)
        if not accessory_data:
            warnings.warn(f"Error: Accessory with ID {accessory_id} not found.")
            return None

        try:
            # Pass only skill_level to the Accessory constructor
            return Accessory(accessory_data, skill_level=skill_level)
        except (ValueError, IndexError) as e:
            warnings.warn(f"Error creating accessory {accessory_id}: {e}")
            return None
