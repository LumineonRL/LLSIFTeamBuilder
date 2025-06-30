from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.simulator import (
    AccessoryManager,
    Deck,
    GameData,
    Guest,
    Play,
    PlayConfig,
    SISManager,
    Song,
    Team,
)

from src.team_builder.env.build_phase import BuildPhase, GameState
from src.team_builder.env.config import EnvConfig
from src.team_builder.env.action_handler import ActionHandler
from src.team_builder.env.observation_manager import ObservationManager
from src.team_builder.env.render import render_human_mode


class LLSIFTeamBuildingEnv(gym.Env):
    """
    A Gymnasium environment for LLSIF team-building

    This environment simulates the process of building a team by selecting
    Cards, Accessories, SIS, a Song, Guest, Accuracy, and Approach Rate
    and returning their simulated score as a reward.
    """

    metadata = {"render_modes": ["human", "agent"], "render_fps": 4}

    # --- Constants for configuration and simulation ---
    DEFAULT_APPROACH_RATE = 9
    MAX_BUILD_PHASE = 4
    DEFAULT_LOG_LEVEL = 50
    SINGLE_TRIAL = 1
    VALID_REWARD_MODES = {"dense", "sparse"}

    def __init__(
        self,
        deck: Deck,
        accessory_manager: AccessoryManager,
        sis_manager: SISManager,
        song: Song,
        game_data: GameData,
        enable_guests: bool,
        accuracy: float,
        data_path: str = "data",
        seed: Optional[int] = None,
        reward_mode: Literal["dense", "sparse"] = "dense",
    ):
        """
        Initializes the team-building environment.

        Args:
            deck: The collection of available cards.
            accessory_manager: The manager for available accessories.
            sis_manager: The manager for available SIS.
            song: The song and difficulty to optimize the team for.
            game_data: Mappings required for the simulator.
            enable_guests: Whether to allow a guest to be chosen.
            accuracy: The percentage chance of hitting a perfect (0.0 to 1.0).
            data_path: The root directory for game mapping files.
            seed: An optional seed for the random number generator.
            reward_mode: 'dense' for rewards at each step, 'sparse' for one final reward.
        """
        super().__init__()
        if reward_mode not in self.VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {self.VALID_REWARD_MODES}, "
                f"got '{reward_mode}'"
            )

        # --- Static Environment Components ---
        self.deck = deck
        self.accessory_manager = accessory_manager
        self.sis_manager = sis_manager
        self.song = song
        self.game_data = game_data
        self.enable_guests = enable_guests
        self.accuracy = accuracy
        self.seed = seed
        self.reward_mode = reward_mode

        data_dir = Path(data_path)
        self.guest_manager = (
            Guest(str(data_dir / "unique_leader_skills.json"))
            if self.enable_guests
            else None
        )

        # --- Configuration and Helper Classes ---
        self.config = EnvConfig(data_path)
        self.action_handler = ActionHandler(self)
        self.obs_manager = ObservationManager(self.config, self)

        self.state: GameState = GameState()

        self.observation_space = self.obs_manager.define_observation_space()
        self.action_space = self.obs_manager.define_action_space()

    def _run_simulation(self, approach_rate: int) -> float:
        """
        Runs the game simulation with the current team and returns the estimated score.

        Args:
            approach_rate: The note approach rate for the simulation (1-10).

        Returns:
            The raw score achieved in a single trial.
        """
        if self.state.team is None:
            return 0.0

        play_config = PlayConfig(
            accuracy=self.accuracy, approach_rate=approach_rate, seed=self.seed
        )
        simulation = Play(self.state.team, self.song, play_config, self.game_data)
        scores = simulation.simulate(
            n_trials=self.SINGLE_TRIAL, log_level=self.DEFAULT_LOG_LEVEL
        )
        return float(scores[0]) if scores else 0.0

    def _is_terminated(self) -> bool:
        """Determines if the episode has reached its terminal state."""
        return self.state.build_phase == BuildPhase.SCORE_SIMULATION

    def _compute_reward(self, terminated: bool, action: int) -> float:
        """
        Calculates the reward for the current step based on the reward mode.
        """
        approach_rate = action + 1 if terminated else self.DEFAULT_APPROACH_RATE
        if terminated:
            self.state.final_approach_rate = approach_rate

        reward = 0.0
        if self.reward_mode == "dense":
            current_score = self._run_simulation(approach_rate)
            reward = current_score - self.state.last_score
            self.state.last_score = current_score
        elif self.reward_mode == "sparse" and terminated:
            reward = self._run_simulation(approach_rate)

        return reward

    def _get_info(self, terminated: bool, raw_action: int) -> Dict[str, Any]:
        """Packages supplementary information for the current step."""
        info = {}
        if terminated:
            info["final_approach_rate"] = self.state.final_approach_rate
            info["raw_final_action"] = raw_action
        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Args:
            seed: An optional seed to reset the environment's random generator.
            options: Optional dictionary to configure the reset.

        Returns:
            A tuple containing the initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed

        team = Team(
            self.deck, self.accessory_manager, self.sis_manager, self.guest_manager
        )
        self.state = GameState(team=team)

        return self.obs_manager.get_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action: An integer representing the action taken by the agent.

        Returns:
            A tuple containing the observation, reward, terminated flag,
            truncated flag (always False), and an info dictionary.
        """
        terminated = self._is_terminated()

        if not terminated:
            self.action_handler.handle_action(action)

        reward = self._compute_reward(terminated, action)

        obs = self.obs_manager.get_obs()
        info = self._get_info(terminated, action)

        return obs, reward, terminated, False, info

    def render(self, mode: str = "human") -> Optional[Union[List[Any], Dict[str, Any]]]:
        """
        Renders the environment's current state.

        Args:
            mode: The rendering mode. 'human' displays a user-friendly summary,
                  'agent' returns a data dictionary or list.

        Returns:
            A data structure if mode is 'agent', otherwise None.
        """
        if mode == "human":
            render_human_mode(self)
            return None
        if mode == "agent":
            return self.obs_manager.get_agent_render_data()
        return None

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask indicating the set of valid actions.

        Returns:
            A numpy array of booleans where True indicates a valid action.
        """
        return self.action_handler.get_action_mask()

    def sample_valid_action(self) -> Optional[int]:
        """
        Samples a random action from the set of currently valid actions.

        Returns:
            A randomly chosen valid action index, or None if no actions are valid.
        """
        valid_actions_mask = self.action_masks()
        valid_action_indices = np.where(valid_actions_mask)[0]
        if len(valid_action_indices) == 0:
            return None
        return self.np_random.choice(valid_action_indices)

    def close(self):
        """Pass."""
        pass
