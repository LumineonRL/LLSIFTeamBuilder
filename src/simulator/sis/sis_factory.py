import json
import warnings
from typing import Dict, List, Optional, Any

from src.simulator.sis.sis_data import SISData
from src.simulator.sis.sis import SIS


class SISFactory:
    """Handles loading SIS data from JSON and creating SIS instances."""

    def __init__(self, sis_json_path: str):
        """Initializes the factory by loading and indexing the SIS data."""
        raw_data = self._load_json(sis_json_path)
        if not isinstance(raw_data, list):
            raise TypeError("SIS data file must be a list of objects.")
        self._sis_data_map: Dict[int, SISData] = self._index_sis_data(raw_data)

    def _load_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Helper to load and parse a JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to load or parse JSON from {json_path}: {e}"
            ) from e

    def _index_sis_data(self, raw_data: List[Dict[str, Any]]) -> Dict[int, SISData]:
        """Converts the raw list of data into a dictionary of SISData objects."""
        indexed_map = {}
        for record in raw_data:
            try:
                sis_id = int(record["id"])
                indexed_map[sis_id] = SISData(**record)
            except (KeyError, TypeError, ValueError) as e:
                warnings.warn(f"Skipping invalid SIS record: {record}. Error: {e}")
        return indexed_map

    def create_sis(self, sid: int) -> Optional[SIS]:
        """Creates an SIS instance by its ID (sid)."""
        sis_data = self._sis_data_map.get(sid)
        if not sis_data:
            warnings.warn(f"Error: SIS with ID {sid} not found.")
            return None
        return SIS(sis_data)
