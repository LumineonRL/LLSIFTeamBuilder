import warnings
from sb3_contrib import MaskablePPO

from src.simulator.card.card_factory import CardFactory
from src.simulator.card.deck import Deck
from src.simulator.accessory.accessory_factory import AccessoryFactory
from src.simulator.accessory.accessory_manager import AccessoryManager
from src.simulator.sis.sis_factory import SISFactory
from src.simulator.sis.sis_manager import SISManager
from src.simulator.song.song_factory import SongFactory
from src.simulator.simulation.game_data import GameData

from src.team_builder.optimizers.rl.trainer import AgentTrainer
from src.team_builder.env.env import LLSIFTeamBuildingEnv
from src.team_builder.optimizers.rl.policy import CustomFeatureExtractor


def main():
    """Main function to configure and start the training."""

    try:
        card_factory = CardFactory(
            cards_json_path="./data/cards.json",
            level_caps_json_path="./data/level_caps.json",
            level_cap_bonuses_path="./data/level_cap_bonuses.json",
        )
        accessory_factory = AccessoryFactory("./data/accessories.json")
        sis_factory = SISFactory("./data/sis.json")
    except RuntimeError as e:
        print(f"FATAL ERROR: Could not load master data files. {e}")
        print(
            "Please ensure cards.json, accessories.json, sis.json, etc., are in the './data' directory."
        )
        return

    deck = Deck(card_factory)
    accessory_manager = AccessoryManager(accessory_factory)
    sis_manager = SISManager(sis_factory)

    deck.load_deck("./data/my_deck.json")
    accessory_manager.load("./data/my_accs.json")
    sis_manager.load("./data/my_sis.json")
    factory = SongFactory("./data/songs.json")
    song = factory.create_song(("HAPPY PARTY TRAIN", "Expert"))
    game_data = GameData("./data/")
    # warnings.filterwarnings("ignore", category=UserWarning)

    env_kwargs = {
        "deck": deck,
        "accessory_manager": accessory_manager,
        "sis_manager": sis_manager,
        "song": song,
        "game_data": game_data,
        "enable_guests": True,
        "accuracy": 0.95,
        "data_path": "./data",
        "reward_mode": "dense",
    }

    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {
            "card_embed_dim": 128,
            "accessory_embed_dim": 64,
            "sis_embed_dim": 32,
            "guest_embed_dim": 32,
            "song_embed_dim": 256,
            "state_embed_dim": 16,
            "attention_heads": 4,
            "dropout": 0.1,
        },
        "net_arch": {"pi": [512, 128], "vf": [512, 128]},
    }

    trainer = AgentTrainer(
        env_class=LLSIFTeamBuildingEnv,
        env_kwargs=env_kwargs,
        log_dir="./logs/llsif_ppo/",
        model_dir="./models/llsif_ppo/",
        n_envs=4,
        algorithm=MaskablePPO,
        policy_kwargs=policy_kwargs,
        device="auto",
        use_action_masking=True,
    )

    trainer.train(total_timesteps=1_000_000, model_save_name="llsif_ppo_agent")


if __name__ == "__main__":
    main()
