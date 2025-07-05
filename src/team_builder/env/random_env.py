import random
import warnings
from typing import Any, Dict

import numpy as np

from src.team_builder.env.env import LLSIFTeamBuildingEnv
from src.simulator import (
    CardFactory,
    AccessoryFactory,
    SISFactory,
    SongFactory,
    Deck,
    AccessoryManager,
    SISManager,
)


class RandomLLSIFTeamBuildingEnv(LLSIFTeamBuildingEnv):
    """
    An extension of the base environment that supports dynamic, randomized
    configurations on reset.
    """

    MAX_CARD_ID = 3911
    MAX_ACC_ID = 549
    MAX_SIS_ID = 546

    def __init__(self, env_kwargs: Dict[str, Any]):
        """
        Initializes the configurable environment.

        Args:
            env_kwargs: A dictionary containing the base environment arguments
                        plus factories and randomization settings.
        """
        self.card_factory: CardFactory = env_kwargs.pop("card_factory")
        self.accessory_factory: AccessoryFactory = env_kwargs.pop("accessory_factory")
        self.sis_factory: SISFactory = env_kwargs.pop("sis_factory")
        self.song_factory: SongFactory = env_kwargs.pop("song_factory")

        self.randomize_on_reset: bool = env_kwargs.pop("randomize_on_reset", True)
        self.reset_frequency: int = env_kwargs.pop("reset_frequency", 1)
        self._reset_counter = 0

        env_kwargs["deck"] = Deck(self.card_factory)
        env_kwargs["accessory_manager"] = AccessoryManager(self.accessory_factory)
        env_kwargs["sis_manager"] = SISManager(self.sis_factory)

        placeholder_song_id = self.song_factory.get_random_identifier()
        if placeholder_song_id is None:
            raise ValueError(
                "SongFactory returned None. Cannot create a placeholder song for the environment."
            )
        env_kwargs["song"] = self.song_factory.create_song(placeholder_song_id)

        super().__init__(**env_kwargs)

    def _randomize_environment(self):
        """Applies a new random configuration to the environment."""
        print("--- Randomizing environment for new episode ---")

        warnings.filterwarnings("ignore", category=UserWarning)

        self.deck.delete_deck()
        all_card_ids = range(1, self.MAX_CARD_ID + 1)
        num_cards = np.random.randint(50, self.MAX_CARD_ID + 1)
        cards_to_add = random.sample(all_card_ids, num_cards)
        for card_id in cards_to_add:
            self.deck.add_card(
                card_id=card_id,
                idolized=bool(np.random.randint(0, 2)),
                level=np.random.randint(100, 501),
                skill_level=np.random.randint(1, 9),
                sis_slots=np.random.randint(1, 9),
            )

        self.accessory_manager.delete()
        all_accessory_ids = range(1, self.MAX_ACC_ID + 1)
        num_accs = np.random.randint(20, self.MAX_ACC_ID + 1)
        accs_to_add = random.sample(all_accessory_ids, num_accs)
        for acc_id in accs_to_add:
            self.accessory_manager.add_accessory(
                accessory_id=acc_id, skill_level=np.random.randint(1, 9)
            )

        self.sis_manager.delete()
        all_sis_ids = range(1, self.MAX_SIS_ID + 1)
        num_sis = np.random.randint(30, self.MAX_SIS_ID + 1)
        sis_to_add = random.sample(all_sis_ids, num_sis)
        for sis_id in sis_to_add:
            self.sis_manager.add_sis(sis_id)

        random_song_id = self.song_factory.get_random_identifier()
        if random_song_id is None:
            raise ValueError(
                "SongFactory returned None. Cannot randomize the song for the environment."
            )
        self.song = self.song_factory.create_song(random_song_id)

        self.accuracy = np.random.beta(8, 2)
        self.enable_guests = bool(np.random.randint(0, 2))

        self.action_space = self.obs_manager.define_action_space()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment. If configured, it will first randomize the
        environment's configuration (song, deck, etc.) before calling the
        parent reset method.
        """
        if self.randomize_on_reset:
            if self._reset_counter % self.reset_frequency == 0:
                self._randomize_environment()
            self._reset_counter += 1

        return super().reset(seed=seed, options=options)
