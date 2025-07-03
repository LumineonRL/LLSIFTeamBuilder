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

        self._consecutive_invalid_actions = 0
        self._max_consecutive_invalid_actions = 2

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

    def _get_final_team_data(self) -> Dict[str, Any]:
        """
        Packages the final team state into a dictionary for logging.
        This captures the team composition at the exact moment the episode ends.
        """
        if self.state.team is None:
            return {
                "team_stats": {"smile": 0, "pure": 0, "cool": 0},
                "final_approach_rate": None,
                "guest": "None",
                "slots": [],
            }

        team = self.state.team
        return {
            "team_stats": {
                "smile": team.total_team_smile,
                "pure": team.total_team_pure,
                "cool": team.total_team_cool,
            },
            "final_approach_rate": self.state.final_approach_rate,
            "guest": self._format_guest_data(team.guest_manager),
            "slots": self._format_slot_data(team.slots),
        }

    def _format_guest_data(self, guest_manager) -> str:
        """Format guest data for logging."""
        if guest_manager and guest_manager.current_guest:
            guest = guest_manager.current_guest
            return (
                f"GuestData(leader_skill_id={guest.leader_skill_id}, "
                f"leader_attribute='{guest.leader_attribute}', "
                f"leader_secondary_attribute={guest.leader_secondary_attribute}, "
                f"leader_value={guest.leader_value}, "
                f"leader_extra_attribute='{guest.leader_extra_attribute}', "
                f"leader_extra_target='{guest.leader_extra_target}', "
                f"leader_extra_value={guest.leader_extra_value})"
            )
        return "None"

    def _format_slot_data(self, slots) -> List[Dict[str, Any]]:
        """Format slot data for logging."""
        slot_data = []
        for i, slot in enumerate(slots):
            if slot.card:
                slot_info = {
                    "slot_number": i + 1,
                    "card_name": slot.card.display_name,
                    "deck_id": slot.card_entry.deck_id if slot.card_entry else None,
                    "stats": {
                        "smile": slot.total_smile,
                        "pure": slot.total_pure,
                        "cool": slot.total_cool,
                    },
                    "accessory": slot.accessory.name if slot.accessory else None,
                    "accessory_id": (
                        slot.accessory_entry.manager_internal_id
                        if slot.accessory_entry
                        else None
                    ),
                    "sis_count": len(slot.sis_entries),
                    "sis_slots_used": sum(sis.sis.slots for sis in slot.sis_entries),
                    "sis_slots_max": slot.card.current_sis_slots if slot.card else 0,
                    "sis_list": [sis.sis.name for sis in slot.sis_entries],
                }
                slot_data.append(slot_info)
        return slot_data

    def _get_info(self, terminated: bool, raw_action: int) -> Dict[str, Any]:
        """Packages supplementary information for the current step."""
        info = {}
        if terminated:
            info["final_approach_rate"] = self.state.final_approach_rate
            info["raw_final_action"] = raw_action
            info["final_team_data"] = self._get_final_team_data()
        return info

    def _validate_action(self, action: int) -> bool:
        """
        Validates if an action is currently valid.
        Returns True if valid, False otherwise.
        """
        action_mask = self.action_masks()
        return 0 <= action < len(action_mask) and action_mask[action]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed

        self._consecutive_invalid_actions = 0

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
        """
        if not self._validate_action(action):
            self._consecutive_invalid_actions += 1
            print(
                f"Invalid action {action} (consecutive: {self._consecutive_invalid_actions})"
            )

            if (
                self._consecutive_invalid_actions
                >= self._max_consecutive_invalid_actions
            ):
                raise RuntimeError(
                    f"Too many consecutive invalid actions ({self._consecutive_invalid_actions}). "
                    f"Last invalid action: {action}. "
                    f"Current phase: {self.state.build_phase.name}. "
                    f"This suggests a problem with action masking implementation."
                )

            obs = self.obs_manager.get_obs()
            info = self._get_info(False, action)
            return obs, 0.0, False, False, info
        else:
            self._consecutive_invalid_actions = 0

        truncated = False

        if self.state.build_phase == BuildPhase.SCORE_SIMULATION:
            final_approach_rate = action + 1
            self.state.final_approach_rate = final_approach_rate

            final_score = self._run_simulation(final_approach_rate)
            if self.reward_mode == "dense":
                reward = final_score - self.state.last_score
            else:  # sparse
                reward = final_score

            terminated = True
            obs = self.obs_manager.get_obs()
            info = self._get_info(terminated, action)

            return obs, reward, terminated, truncated, info

        self.action_handler.handle_action(action)

        reward = 0.0
        if self.reward_mode == "dense":
            current_score = self._run_simulation(self.DEFAULT_APPROACH_RATE)
            reward = current_score - self.state.last_score
            self.state.last_score = current_score

        terminated = False
        obs = self.obs_manager.get_obs()
        info = self._get_info(terminated, action)

        return obs, reward, terminated, truncated, info

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
