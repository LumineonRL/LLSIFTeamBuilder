from pathlib import Path
from typing import Type, Dict, Any, Optional

import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import (
    MaskableEvalCallback,
)
from sb3_contrib.common.maskable.evaluation import (
    evaluate_policy,
)

from src.team_builder.env.env import LLSIFTeamBuildingEnv
from src.team_builder.optimizers.rl.util.tensorboard_logger import TeamLogCallback


class AgentTrainer:
    """
    Manages the training of a Stable Baselines 3 agent for LLSIF team building.
    This version is updated to support MaskablePPO for proper action masking.
    """

    def __init__(
        self,
        env_class: Type[LLSIFTeamBuildingEnv],
        env_kwargs: Dict[str, Any],
        log_dir: str = "./logs/",
        model_dir: str = "./models/",
        n_envs: int = 4,
        algorithm: Type[BaseAlgorithm] = MaskablePPO,
        policy: str = "MultiInputPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        use_action_masking: bool = True,
    ):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.algorithm = algorithm
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.device = self._get_device(device)
        self.use_action_masking = use_action_masking
        self.n_envs = n_envs

        self.hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        vec_env_cls = (
            DummyVecEnv
            if self.use_action_masking
            else (SubprocVecEnv if self.n_envs > 1 else DummyVecEnv)
        )
        self.vec_env = make_vec_env(
            self._make_env,
            n_envs=self.n_envs,
            vec_env_cls=vec_env_cls,
        )

        self.model = self._setup_model()

    def _make_env(self):
        """Factory function to create a single environment instance."""
        base_env = self.env_class(**self.env_kwargs)
        if self.use_action_masking:

            def mask_fn(env):
                actual_env = env.unwrapped if hasattr(env, "unwrapped") else env
                return actual_env.action_masks()

            return ActionMasker(base_env, mask_fn)
        return base_env

    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_model(self, model_path: Optional[str] = None) -> BaseAlgorithm:
        """Sets up the SB3 model, loading or creating a new one."""
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}...")
            return self.algorithm.load(model_path, env=self.vec_env, device=self.device)

        print("Creating a new model with the following hyperparameters:")
        for key, val in self.hyperparams.items():
            print(f"  {key}: {val}")

        return self.algorithm(
            policy=self.policy,
            env=self.vec_env,
            policy_kwargs=self.policy_kwargs,
            verbose=1,
            device=self.device,
            tensorboard_log=str(self.log_dir),
            **self.hyperparams,
        )

    def train(self, total_timesteps: int, model_save_name: str):
        """Starts the training process with evaluation and logging."""
        print(
            f"\nStarting training on device: {self.device} with {self.n_envs} environments."
        )
        if self.use_action_masking:
            print("Action masking is ENABLED.")

        team_log_callback = TeamLogCallback(verbose=1)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // self.n_envs, 1),
            save_path=str(self.model_dir),
            name_prefix=f"{model_save_name}_checkpoint",
        )

        eval_env = make_vec_env(self._make_env, n_envs=1, vec_env_cls=DummyVecEnv)

        if self.use_action_masking:
            eval_callback = MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir),
                log_path=str(self.log_dir),
                eval_freq=max(10000 // self.n_envs, 1),
                n_eval_episodes=20,
                deterministic=True,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir),
                log_path=str(self.log_dir),
                eval_freq=max(10000 // self.n_envs, 1),
                n_eval_episodes=20,
                deterministic=True,
            )

        callbacks = [team_log_callback, checkpoint_callback, eval_callback]

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
            final_model_path = self.model_dir / f"{model_save_name}_final.zip"
            self.model.save(final_model_path)
            print(f"Training finished. Final model saved to {final_model_path}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            interrupt_model_path = self.model_dir / f"{model_save_name}_interrupted.zip"
            self.model.save(interrupt_model_path)
            print(f"Model saved to {interrupt_model_path}")
        finally:
            eval_env.close()
