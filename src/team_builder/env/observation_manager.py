from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable, Sequence

import numpy as np
from gymnasium import spaces

from src.simulator import Card, Team
from src.team_builder.env.build_phase import BuildPhase

if TYPE_CHECKING:
    from .config import EnvConfig
    from .env import LLSIFTeamBuildingEnv


class ObservationManager:
    """
    Manages observation space definition, action space definition, and state
    serialization for the environment.
    """

    APPROACH_RATE_CHOICES: int = 10
    ACTION_ID_OFFSET: int = 2

    def __init__(self, config: "EnvConfig", env: "LLSIFTeamBuildingEnv"):
        """
        Initializes the ObservationManager.

        Args:
            config: The environment configuration object.
            env: The main environment instance.
        """
        self.config = config
        self.env = env
        self.card_feature_size = self._calculate_card_feature_size()

    # --- Action Space Definition ---

    def define_action_space(self) -> spaces.Discrete:
        """
        Defines the action space for the environment by finding the maximum
        number of actions possible across all build phases.

        In other words, find whether the Deck, Accessory Manager, or SIS
        Manager contains the most entries.
        """
        max_actions = max(
            self._get_max_deck_actions(),
            self._get_max_accessory_actions(),
            self._get_max_sis_actions(),
            self._get_max_guest_actions(),
            self.APPROACH_RATE_CHOICES,
        )
        return spaces.Discrete(max_actions)

    def _get_max_deck_actions(self) -> int:
        """Calculates the number of actions for the deck selection phase."""
        if not self.env.deck.entries:
            return 0
        return max(self.env.deck.entries.keys()) + 1

    def _get_max_accessory_actions(self) -> int:
        """Calculates actions for the accessory phase (items + pass)."""
        if not self.env.accessory_manager.accessories:
            return self.ACTION_ID_OFFSET
        max_id = max(self.env.accessory_manager.accessories.keys())
        return max_id + self.ACTION_ID_OFFSET

    def _get_max_sis_actions(self) -> int:
        """Calculates actions for the SIS phase (items + pass)."""
        if not self.env.sis_manager.skills:
            return self.ACTION_ID_OFFSET
        max_id = max(self.env.sis_manager.skills.keys())
        return max_id + self.ACTION_ID_OFFSET

    def _get_max_guest_actions(self) -> int:
        """Calculates actions for the guest selection phase."""
        if self.env.enable_guests and self.env.guest_manager:
            return len(self.env.guest_manager.all_guests)
        # If guests are disabled, there's effectively one 'no-op' or default choice.
        return 1

    # --- Observation Space Definition ---

    def define_observation_space(self) -> spaces.Dict:
        """Defines the observation space for the environment."""
        return spaces.Dict(
            {
                "deck": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.MAX_CARDS_IN_DECK, self.card_feature_size),
                    dtype=np.float32,
                ),
                # TODO: Implement other observation components (team, accessories, SIS)
                "build_phase": spaces.Discrete(len(BuildPhase)),
                "current_slot": spaces.Discrete(Team.NUM_SLOTS),
            }
        )

    def get_obs(self) -> Dict[str, Any]:
        """
        Constructs the current observation from the environment state.

        Returns:
            A dictionary containing the serialized state:
                - 'deck': Serialized deck information as a numpy array.
                - 'build_phase': The current build phase index (0-4).
                - 'current_slot': The current team slot index being filled.

        Raises:
            ValueError: If the deck contains invalid data or fails serialization.
        """
        deck_obs = np.zeros(
            (self.config.MAX_CARDS_IN_DECK, self.card_feature_size),
            dtype=np.float32,
        )
        sorted_deck_entries = sorted(
            self.env.deck.entries.values(), key=lambda x: x.deck_id
        )

        for i, entry in enumerate(sorted_deck_entries):
            if i >= self.config.MAX_CARDS_IN_DECK:
                break
            deck_obs[i] = self._serialize_card(entry.card)

        return {
            "deck": deck_obs,
            "build_phase": int(self.env.state.build_phase),
            "current_slot": self.env.state.current_slot_idx,
        }

    # --- Agent Rendering ---

    def get_agent_render_data(self) -> List:
        """
        Returns data structures for the 'agent' render mode based on the
        current build phase, using a strategy pattern for clarity.
        """
        render_strategies: Dict[BuildPhase, Callable[[], List]] = {
            BuildPhase.CARD_SELECTION: self._render_deck_phase,
            BuildPhase.ACCESSORY_ASSIGNMENT: self._render_accessory_phase,
            BuildPhase.SIS_ASSIGNMENT: self._render_sis_phase,
            BuildPhase.GUEST_SELECTION: self._render_guest_phase,
            BuildPhase.SCORE_SIMULATION: self._render_approach_rate_phase,
        }
        strategy = render_strategies.get(self.env.state.build_phase)
        return strategy() if strategy else []

    def _render_deck_phase(self) -> List:
        """Provides rendering data for the deck selection phase."""
        unassigned_ids = self._get_unassigned_deck_ids()
        return [self.env.deck.get_entry(did) for did in unassigned_ids]

    def _render_accessory_phase(self) -> List:
        """Provides rendering data for the accessory selection phase."""
        if self.env.state.team is None:
            return ["Pass"]
        return ["Pass"] + self.env.accessory_manager.get_unassigned_accessories(
            self.env.state.team.assigned_accessory_ids
        )

    def _render_sis_phase(self) -> List:
        """Provides rendering data for the School Idol Skill selection phase."""
        if self.env.state.team is None:
            return ["Pass"]
        return ["Pass"] + self.env.sis_manager.get_unassigned_sis(
            self.env.state.team.assigned_sis_ids
        )

    def _render_guest_phase(self) -> List:
        """Provides rendering data for the guest selection phase."""
        if self.env.enable_guests and self.env.guest_manager:
            return list(self.env.guest_manager.all_guests.values())
        return ["No Guests Enabled"]

    def _render_approach_rate_phase(self) -> List:
        """Provides rendering data for the approach rate selection phase."""
        return list(range(1, self.APPROACH_RATE_CHOICES + 1))

    # --- Serialization and Helper Methods ---

    def _calculate_card_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single card."""
        c = self.config
        parts = [
            1,  # Rarity
            len(c.ATTRIBUTE_MAP),  # Attribute
            len(c.character_map) + 1,  # Character
            3,  # Stats (Smile, Pure, Cool)
            1, # SIS Slots
            len(c.LEADER_ATTRIBUTE_MAP),  # LS Attribute
            len(c.LEADER_ATTRIBUTE_MAP),  # LS Secondary Attribute
            1,  # LS Value
            len(c.LEADER_ATTRIBUTE_MAP),  # LS Extra Attribute
            len(c.character_map) + 1,  # LS Extra Target
            1,  # LS Extra Value
            1,  # Skill level
            len(c.skill_type_map),  # Skill Type
            len(c.skill_activation_map),  # Skill Activation
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Thresholds
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Chances
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Values
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Durations
        ]
        return sum(parts)

    def _get_unassigned_deck_ids(self) -> List[int]:
        """Helper to get a sorted list of deck IDs not assigned to the team."""
        if self.env.state.team is None:
            return []
        all_deck_ids = set(self.env.deck.entries.keys())
        unassigned_ids = all_deck_ids - self.env.state.team.assigned_deck_ids
        return sorted(list(unassigned_ids))

    def _serialize_card(self, card: Card) -> np.ndarray:
        """
        Converts a Card object into a normalized, flat numpy array.

        Raises:
            ValueError: If the card is None or an error occurs during serialization.
        """
        if card is None:
            raise ValueError("Input card cannot be None.")

        try:
            features = []
            c = self.config
            ls = card.leader_skill
            skill = card.skill
            ls_attr_map = c.LEADER_ATTRIBUTE_MAP

            # --- Basic Features ---
            features.append(c.RARITY_MAP.get(card.rarity, 0) / (len(c.RARITY_MAP) - 1))
            features.extend(self._one_hot(card.attribute, c.ATTRIBUTE_MAP))
            features.extend(self._one_hot_character(card))
            features.append(card.stats.smile / c.MAX_STAT_VALUE)
            features.append(card.stats.pure / c.MAX_STAT_VALUE)
            features.append(card.stats.cool / c.MAX_STAT_VALUE)
            features.append(card.current_sis_slots / c.MAX_SIS_SLOTS)

            # # --- Leader Skill ---
            features.extend(self._one_hot(ls.attribute, ls_attr_map, "None"))
            features.extend(self._one_hot(ls.secondary_attribute, ls_attr_map, "None"))
            features.append((ls.value or 0.0) / c.MAX_LS_VALUE)
            features.extend(self._one_hot(ls.extra_attribute, ls_attr_map, "None"))
            if ls.extra_target is not None:
                extra_target_vector = c.ls_extra_target_map.get(
                    ls.extra_target, c.ls_extra_target_default_vector
                )
            else:
                extra_target_vector = c.ls_extra_target_default_vector
            features.extend(extra_target_vector)
            features.append((ls.extra_value or 0.0) / c.MAX_LS_VALUE)

            # # --- Card Skill ---
            features.append((card.current_skill_level or 0) / c.MAX_SKILL_LEVEL)
            features.extend(self._one_hot(skill.type, c.skill_type_map))
            features.extend(self._one_hot(skill.activation, c.skill_activation_map))

            # --- Padded Skill Lists ---
            features.extend(
                self._pad_and_normalize(
                    skill.thresholds, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_THRESHOLD
                )
            )
            features.extend(
                self._pad_and_normalize(skill.chances, c.MAX_SKILL_LIST_ENTRIES, 1.0)
            )
            features.extend(
                self._pad_and_normalize(
                    skill.values, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_VALUE
                )
            )
            features.extend(
                self._pad_and_normalize(
                    skill.durations, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_DURATION
                )
            )

            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize card_id={card.card_id}: {e}") from e

    def _one_hot(
        self,
        value: Optional[str],
        mapping: Dict[str, int],
        default: Optional[str] = None,
    ) -> np.ndarray:
        """Creates a one-hot encoded vector from a value and a mapping dict."""
        one_hot = np.zeros(len(mapping), dtype=np.float32)
        key = value or default
        if key is not None:
            idx = mapping.get(key)
            if idx is not None:
                one_hot[idx] = 1.0
        return one_hot

    def _one_hot_character(self, card: Card) -> np.ndarray:
        """
        Creates a one-hot encoded vector for the card's character.

        If the character is not in the main list defined in the config,
        it is mapped to the 'other' category at index 0.
        """
        c = self.config

        num_total_characters = len(c.character_map) + 1
        one_hot = np.zeros(num_total_characters, dtype=np.float32)

        index = c.character_map.get(card.character, c.character_other_index)

        one_hot[index] = 1.0
        return one_hot

    @staticmethod
    def _pad_and_normalize(
        data: Optional[Sequence[Union[int, float]]], max_len: int, norm_factor: float
    ) -> np.ndarray:
        """
        Returns a zero-padded, normalized array from an input list, clipped to max_len.
        """
        padded = np.zeros(max_len, dtype=np.float32)
        if not data:
            return padded

        data_cleaned = [x for x in data if x is not None][:max_len]
        if data_cleaned:
            padded[: len(data_cleaned)] = (
                np.array(data_cleaned, dtype=np.float32) / norm_factor
            )
        return padded
