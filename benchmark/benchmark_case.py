import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.simulator.card.card_factory import CardFactory
from src.simulator.card.deck import Deck
from src.simulator.accessory.accessory_factory import AccessoryFactory
from src.simulator.accessory.accessory_manager import AccessoryManager
from src.simulator.sis.sis_factory import SISFactory
from src.simulator.sis.sis_manager import SISManager
from src.simulator.song.song_factory import SongFactory
from src.simulator.simulation.game_data import GameData
from src.team_builder.env.env import LLSIFTeamBuildingEnv


@dataclass
class BenchmarkCaseConfig:
    """Configuration for a single benchmark case, loaded from its config.json."""

    song_name: str
    song_difficulty: str
    accuracy: float = 0.95
    enable_guests: bool = True
    description: Optional[str] = None


class BenchmarkCase:
    """
    Represents a single, self-contained benchmark case with all necessary game data loaded.
    This object encapsulates one benchmarking scenario.
    """

    def __init__(self, case_path: Path, master_data_path: Path = Path("./data")):
        self.path = case_path
        self.name = case_path.name
        self.master_data_path = master_data_path

        self.config = self._load_config()
        self._init_factories()

        self.deck = self._load_deck()
        self.accessory_manager = self._load_accessories()
        self.sis_manager = self._load_sis()
        self.song = self.song_factory.create_song(
            (self.config.song_name, self.config.song_difficulty)
        )
        self.game_data = GameData(str(master_data_path))

    def _load_config(self) -> BenchmarkCaseConfig:
        """Load benchmark case configuration from config.json."""
        config_path = self.path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found in benchmark case: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BenchmarkCaseConfig(**data)

    def _init_factories(self):
        """Initialize all factories from master data."""
        self.card_factory = CardFactory(
            cards_json_path=str(self.master_data_path / "cards.json"),
            level_caps_json_path=str(self.master_data_path / "level_caps.json"),
            level_cap_bonuses_path=str(
                self.master_data_path / "level_cap_bonuses.json"
            ),
        )
        self.accessory_factory = AccessoryFactory(
            str(self.master_data_path / "accessories.json")
        )
        self.sis_factory = SISFactory(str(self.master_data_path / "sis.json"))
        self.song_factory = SongFactory(str(self.master_data_path / "songs.json"))

    def _load_deck(self) -> Deck:
        """Load deck from the benchmark case's deck.json."""
        deck = Deck(self.card_factory)
        deck.load_deck(str(self.path / "deck.json"))
        return deck

    def _load_accessories(self) -> AccessoryManager:
        """Load accessories from the benchmark case's accs.json."""
        manager = AccessoryManager(self.accessory_factory)
        manager.load(str(self.path / "accs.json"))
        return manager

    def _load_sis(self) -> SISManager:
        """Load SIS from the benchmark case's my_sis.json."""
        manager = SISManager(self.sis_factory)
        manager.load(str(self.path / "sis.json"))
        return manager

    def get_env(self) -> LLSIFTeamBuildingEnv:
        """Get a configured environment for this benchmark case."""
        env_kwargs = {
            "deck": self.deck,
            "accessory_manager": self.accessory_manager,
            "sis_manager": self.sis_manager,
            "song": self.song,
            "game_data": self.game_data,
            "enable_guests": self.config.enable_guests,
            "accuracy": self.config.accuracy,
            "data_path": str(self.master_data_path),
            "reward_mode": "dense",
        }
        return LLSIFTeamBuildingEnv(**env_kwargs)
