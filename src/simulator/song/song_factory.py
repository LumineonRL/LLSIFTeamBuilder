import json
import random
import warnings
from typing import Dict, Optional, Any, Union, Tuple

from src.simulator.song.song_data import SongData
from src.simulator.song.song import Song


class SongFactory:
    """
    Handles loading song data from JSON and creating Song instances
    using either a song ID or a (title, difficulty) tuple.
    """

    def __init__(self, songs_json_path: str):
        raw_data = self._load_json(songs_json_path)
        if not isinstance(raw_data, dict):
            raise TypeError("Songs data file must be a dictionary of objects.")

        # Create two lookup maps for flexible creation
        self._song_data_by_id: Dict[str, SongData] = {}
        self._song_data_by_title_diff: Dict[Tuple[str, str], SongData] = {}
        self._index_song_data(raw_data)

    def _load_json(self, json_path: str) -> Dict:
        """Helper to load and parse a JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to load or parse JSON from {json_path}: {e}"
            ) from e

    def _index_song_data(self, raw_data: Dict[str, Any]) -> None:
        """Converts the raw dictionary data into SongData objects and indexes them."""
        for song_id_json, record in raw_data.items():
            try:
                # Strip '.json' from the key to get the clean song_id
                song_id = song_id_json.replace(".json", "")

                data_instance = SongData(
                    song_id=song_id,
                    title=record.get("title", "Unknown Title"),
                    difficulty=record.get("difficulty", "Unknown"),
                    group=record.get("group", "Unknown"),
                    attribute=record.get("attribute", "Unknown"),
                    notes=record.get("notes", []),
                )

                # Index by song_id
                self._song_data_by_id[song_id] = data_instance

                # Index by (title, difficulty) tuple
                title_diff_key = (data_instance.title, data_instance.difficulty)
                self._song_data_by_title_diff[title_diff_key] = data_instance

            except (ValueError, TypeError) as e:
                warnings.warn(
                    f"Warning: Skipping invalid song record with key '{song_id_json}': {e}"
                )

    def create_song(self, identifier: Union[str, Tuple[str, str]]) -> Optional[Song]:
        """
        Creates a configured, stateful Song instance.

        Args:
            identifier: Either the song_id (str) or a tuple of (title, difficulty).

        Returns:
            A Song object or None if not found.
        """
        song_data = None
        if isinstance(identifier, str):
            song_data = self._song_data_by_id.get(identifier)
        elif isinstance(identifier, tuple) and len(identifier) == 2:
            song_data = self._song_data_by_title_diff.get(identifier)
        else:
            warnings.warn(
                f"Invalid identifier type: {type(identifier)}. Must be str or (str, str) tuple."
            )
            return None

        if not song_data:
            warnings.warn(f"Error: Song with identifier '{identifier}' not found.")
            return None

        return Song(song_data)

    def get_random_identifier(self) -> Optional[str]:
        """Returns a random song_id from the loaded songs."""
        if not self._song_data_by_id:
            return None
        all_ids = list(self._song_data_by_id.keys())
        return random.choice(all_ids)
