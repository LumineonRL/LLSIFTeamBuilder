import warnings
from sb3_contrib import MaskablePPO

from src.simulator.card.card_factory import CardFactory
from src.simulator.accessory.accessory_factory import AccessoryFactory
from src.simulator.sis.sis_factory import SISFactory
from src.simulator.song.song_factory import SongFactory
from src.simulator.simulation.game_data import GameData

from src.team_builder.optimizers.rl.trainer import AgentTrainer
from src.team_builder.env.random_env import RandomLLSIFTeamBuildingEnv
from src.team_builder.optimizers.rl.policy import CustomFeatureExtractor


def main():
    """Main function to configure and start the training for a generalist agent."""

    try:
        card_factory = CardFactory(
            cards_json_path="./data/cards.json",
            level_caps_json_path="./data/level_caps.json",
            level_cap_bonuses_path="./data/level_cap_bonuses.json",
        )
        accessory_factory = AccessoryFactory("./data/accessories.json")
        sis_factory = SISFactory("./data/sis.json")
        song_factory = SongFactory("./data/songs.json")
    except RuntimeError as e:
        print(f"FATAL ERROR: Could not load master data files. {e}")
        print(
            "Please ensure cards.json, accessories.json, sis.json, etc., are in the './data' directory."
        )
        return

    game_data = GameData("./data/")
    warnings.filterwarnings("ignore", category=UserWarning)

    env_kwargs = {
        "card_factory": card_factory,
        "accessory_factory": accessory_factory,
        "sis_factory": sis_factory,
        "song_factory": song_factory,
        "game_data": game_data,
        "data_path": "./data",
        "reward_mode": "dense",
        "randomize_on_reset": True,
        "reset_frequency": 1028,
        "enable_guests": True,
        "accuracy": 0.95,
    }

    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {
            "card_embed_dim": 256,
            "accessory_embed_dim": 128,
            "sis_embed_dim": 64,
            "guest_embed_dim": 64,
            "song_embed_dim": 512,
            "state_embed_dim": 32,
            "attention_heads": 8,
            "attention_layers": 2,
            "dropout": 0.1,
        },
        "net_arch": {"pi": [1024, 512], "vf": [1024, 512]},
    }

    trainer = AgentTrainer(
        env_class=RandomLLSIFTeamBuildingEnv,
        env_kwargs={"env_kwargs": env_kwargs},
        log_dir="./logs/llsif_ppo_generalist/",
        model_dir="./models/llsif_ppo_generalist/",
        n_envs=4,
        algorithm=MaskablePPO,
        policy_kwargs=policy_kwargs,
        device="auto",
        use_action_masking=True,
        learning_rate=1e-4,
        ent_coef=0.01,
    )

    trainer.train(total_timesteps=1_000_000, model_save_name="llsif_ppo_agent_generalist")


if __name__ == "__main__":
    main()
