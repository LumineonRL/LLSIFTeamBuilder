from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable, Sequence

import numpy as np
from gymnasium import spaces

from src.simulator import Accessory, Card, Team, SIS
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
        self.accessory_feature_size = self._calculate_accessory_feature_size()
        self.sis_feature_size = self._calculate_sis_feature_size()
        self.team_feature_size = 3 + Team.NUM_SLOTS

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
        return 1

    # --- Observation Space Definition ---

    def define_observation_space(self) -> spaces.Dict:
        """Defines the observation space for the environment."""
        team_feature_size = 3 + Team.NUM_SLOTS  # 3 for total stats, 9 for SIS slots
        return spaces.Dict(
            {
                "deck": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.MAX_CARDS_IN_DECK, self.card_feature_size),
                    dtype=np.float32,
                ),
                "team": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(team_feature_size,),
                    dtype=np.float32,
                ),
                "accessories": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.config.MAX_ACCESSORIES_IN_INVENTORY,
                        self.accessory_feature_size,
                    ),
                    dtype=np.float32,
                ),
                "sis": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.config.MAX_SIS_IN_INVENTORY,
                        self.sis_feature_size,
                    ),
                    dtype=np.float32,
                ),
                # TODO: Add Guest and Song
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
                - 'team': Serialized team information as a numpy array.
                - 'accessories': Serialized accessory info as a numpy array.
                - 'sis': Serialized SIS info as a numpy array.
                - 'build_phase': The current build phase index (0-4).
                - 'current_slot': The current team slot index being filled.

        Raises:
            ValueError: If the deck contains invalid data or fails serialization.
        """
        # Deck observation
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

        # Team observation
        team_obs = self._serialize_team(self.env.state.team)

        # Accessory observation
        accessories_obs = np.zeros(
            (
                self.config.MAX_ACCESSORIES_IN_INVENTORY,
                self.accessory_feature_size,
            ),
            dtype=np.float32,
        )
        sorted_accessories = sorted(
            self.env.accessory_manager.accessories.values(),
            key=lambda pa: pa.manager_internal_id,
        )
        for i, player_accessory in enumerate(sorted_accessories):
            if i >= self.config.MAX_ACCESSORIES_IN_INVENTORY:
                break
            accessories_obs[i] = self._serialize_accessory(player_accessory.accessory)

        # SIS observation
        sis_obs = np.zeros(
            (
                self.config.MAX_SIS_IN_INVENTORY,
                self.sis_feature_size,
            ),
            dtype=np.float32,
        )
        sorted_sis = sorted(
            self.env.sis_manager.skills.values(),
            key=lambda ps: ps.manager_internal_id,
        )
        for i, player_sis in enumerate(sorted_sis):
            if i >= self.config.MAX_SIS_IN_INVENTORY:
                break
            sis_obs[i] = self._serialize_sis(player_sis.sis)

        return {
            "deck": deck_obs,
            "team": team_obs,
            "accessories": accessories_obs,
            "sis": sis_obs,
            "build_phase": int(self.env.state.build_phase),
            "current_slot": self.env.state.current_slot_idx,
        }

    # --- Agent Rendering ---

    def get_agent_render_data(self) -> List:
        """
        Returns data structures for the 'agent' render mode based on the
        current build phase.
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
            1,  # SIS Slots
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
            len(c.character_map) + 1,  # Skill Target
        ]
        return sum(parts)

    def _calculate_accessory_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single accessory."""
        c = self.config
        parts = [
            len(c.character_map) + 1,  # Character
            1,  # Has card_id
            3,  # Stats (Smile, Pure, Cool)
            1,  # Skill Level
            len(c.ACCESSORY_SKILL_TARGET_MAP),  # Skill Target
            len(c.ACCESSORY_SKILL_TYPE_MAP),  # Skill Type
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Chances
            1,  # Skill Threshold
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Durations
            c.MAX_SKILL_LIST_ENTRIES,  # Skill Values
        ]
        return sum(parts)

    def _calculate_sis_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single SIS."""
        c = self.config
        num_chars_and_other = len(c.character_map) + 1
        num_attributes = len(c.ATTRIBUTE_MAP)
        parts = [
            len(c.SIS_EFFECT_MAP),  # effect (one-hot)
            1,  # slots
            num_attributes,  # attribute (one-hot)
            len(c.SIS_GROUP_MAP),  # group (one-hot)
            num_chars_and_other + num_attributes,  # equip_restriction (multi-hot)
            1,  # target (binary)
            1,  # value (normalized)
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

            # --- Leader Skill ---
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

            # --- Card Skill ---
            features.append((card.current_skill_level or 0) / c.MAX_SKILL_LEVEL)
            features.extend(self._one_hot(skill.type, c.skill_type_map))
            features.extend(self._one_hot(skill.activation, c.skill_activation_map))

            if skill.activation == "Score":
                threshold_norm_factor = c.MAX_SKILL_THRESHOLD_SCORE
            elif skill.activation == "Time":
                threshold_norm_factor = c.MAX_SKILL_THRESHOLD_TIME
            else:
                threshold_norm_factor = c.MAX_SKILL_THRESHOLD_DEFAULT
            features.extend(
                self._pad_and_normalize(
                    skill.thresholds, c.MAX_SKILL_LIST_ENTRIES, threshold_norm_factor
                )
            )

            print(skill.chances)

            features.extend(
                self._pad_and_normalize(skill.chances, c.MAX_SKILL_LIST_ENTRIES, 1.0)
            )

            if skill.type in ["Appeal Boost", "Skill Rate Up"]:
                value_norm_factor = c.MAX_SKILL_VALUE_PERCENT
            elif skill.type == "Amplify":
                value_norm_factor = c.MAX_SKILL_VALUE_AMP
            elif skill.type == "Healer":
                value_norm_factor = c.MAX_SKILL_VALUE_HEAL
            elif skill.type in ["Combo Bonus Up", "Perfect Score Up"]:
                value_norm_factor = c.MAX_SKILL_VALUE_FLAT
            else:  # "Scorer" and any other default
                value_norm_factor = c.MAX_SKILL_VALUE_DEFAULT
            features.extend(
                self._pad_and_normalize(
                    skill.values, c.MAX_SKILL_LIST_ENTRIES, value_norm_factor
                )
            )

            features.extend(
                self._pad_and_normalize(
                    skill.durations, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_DURATION
                )
            )

            if skill.target and skill.target.strip():
                target_vector = c.skill_target_map.get(
                    skill.target, c.skill_target_default_vector
                )
            else:
                target_vector = c.skill_target_default_vector
            features.extend(target_vector)

            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize card_id={card.card_id}: {e}") from e

    def _serialize_accessory(self, accessory: Accessory) -> np.ndarray:
        """
        Converts an Accessory object into a normalized, flat numpy array.

        Raises:
            ValueError: If the accessory is None or an error occurs during serialization.
        """
        if accessory is None:
            raise ValueError("Input accessory cannot be None.")

        try:
            features = []
            c = self.config
            skill = accessory.skill

            # --- Basic Features ---
            num_total_characters = len(c.character_map) + 1
            char_one_hot = np.zeros(num_total_characters, dtype=np.float32)
            char_index = c.character_map.get(
                accessory.character, c.character_other_index
            )
            char_one_hot[char_index] = 1.0
            features.extend(char_one_hot)

            features.append(1.0 if accessory.card_id else 0.0)
            features.append(accessory.stats.smile / c.MAX_ACCESSORY_STAT)
            features.append(accessory.stats.pure / c.MAX_ACCESSORY_STAT)
            features.append(accessory.stats.cool / c.MAX_ACCESSORY_STAT)

            # # --- Skill ---
            features.append(accessory.skill_level / c.MAX_SKILL_LEVEL)

            # # Skill Target
            num_targets = len(c.ACCESSORY_SKILL_TARGET_MAP)
            target_vector = np.zeros(num_targets, dtype=np.float32)
            if skill.target == "All":
                target_vector = np.ones(num_targets, dtype=np.float32)
            elif skill.target and skill.target in c.ACCESSORY_SKILL_TARGET_MAP:
                idx = c.ACCESSORY_SKILL_TARGET_MAP[skill.target]
                target_vector[idx] = 1.0
            features.extend(target_vector)

            features.extend(self._one_hot(skill.type, c.ACCESSORY_SKILL_TYPE_MAP))

            modified_chances = [chance / 100 for chance in skill.chances]

            features.extend(
                self._pad_and_normalize(modified_chances, c.MAX_SKILL_LIST_ENTRIES, 1.0)
            )

            features.append(
                (accessory.skill_threshold or 0.0) / c.MAX_ACC_SKILL_THRESHOLD
            )

            features.extend(
                self._pad_and_normalize(
                    skill.durations, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_DURATION
                )
            )

            if skill.type in ["Appeal Boost", "Skill Rate Up"]:
                skill_value = [values / 100 for values in skill.values]
                value_norm_factor = c.MAX_SKILL_VALUE_PERCENT
            elif skill.type == "Amplify":
                skill_value = skill.values
                value_norm_factor = c.MAX_SKILL_VALUE_AMP
            elif skill.type == "Healer":
                skill_value = skill.values
                value_norm_factor = c.MAX_SKILL_VALUE_HEAL
            elif skill.type in ["Combo Bonus Up", "Perfect Score Up", "Spark"]:
                skill_value = skill.values
                value_norm_factor = c.MAX_SKILL_VALUE_FLAT
            else:  # "Scorer" and any other default
                skill_value = skill.values
                value_norm_factor = c.MAX_SKILL_VALUE_DEFAULT
            features.extend(
                self._pad_and_normalize(
                    skill_value, c.MAX_SKILL_LIST_ENTRIES, value_norm_factor
                )
            )

            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Failed to serialize accessory_id={accessory.accessory_id}: {e}"
            ) from e

    def _serialize_sis(self, sis: SIS) -> np.ndarray:
        """
        Converts a SIS object into a normalized, flat numpy array.

        Raises:
            ValueError: If the sis is None or an error occurs during serialization.
        """
        if sis is None:
            raise ValueError("Input SIS cannot be None.")

        try:
            features = []
            c = self.config

            # Effect (one-hot)
            features.extend(self._one_hot(sis.effect, c.SIS_EFFECT_MAP))

            # Slots (normalized)
            features.append(sis.slots / c.MAX_SIS_SLOTS)

            # Attribute (one-hot)
            features.extend(self._one_hot(sis.attribute, c.ATTRIBUTE_MAP))

            # Group (one-hot, handles '' or None as all-zeros)
            features.extend(self._one_hot(sis.group, c.SIS_GROUP_MAP))

            # Equip Restriction (multi-hot)
            restriction = sis.equip_restriction or ""
            restriction_vector = c.sis_equip_restriction_map.get(
                restriction, c.sis_equip_restriction_default_vector
            )
            features.extend(restriction_vector)

            # Target (binary: 0.0 for 'self', 1.0 for 'all')
            features.append(1.0 if sis.target == "all" else 0.0)

            # Value (normalized by effect type)
            norm_factor = 1.0
            if sis.effect == "all percent boost":
                norm_factor = c.MAX_SIS_VALUE_ALL_PERCENT
            elif sis.effect == "charm":
                norm_factor = c.MAX_SIS_VALUE_CHARM
            elif sis.effect == "heal":
                norm_factor = c.MAX_SIS_VALUE_HEAL
            elif sis.effect == "self flat boost":
                norm_factor = c.MAX_SIS_VALUE_SELF_FLAT
            elif sis.effect == "self percent boost":
                norm_factor = c.MAX_SIS_VALUE_SELF_PERCENT
            elif sis.effect == "trick":
                norm_factor = c.MAX_SIS_VALUE_TRICK

            normalized_value = (
                (sis.value or 0.0) / norm_factor if norm_factor > 0 else 0.0
            )
            features.append(normalized_value)

            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize sis_id={sis.id}: {e}") from e

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

    def _serialize_team(self, team: Optional[Team]) -> np.ndarray:
        """Converts the Team object into a normalized, flat numpy array."""
        team_obs = np.zeros(self.team_feature_size, dtype=np.float32)
        if not team:
            return team_obs

        # First 3 features: Total team stats
        team_obs[0] = team.total_team_smile / self.config.MAX_TEAM_STAT_VALUE
        team_obs[1] = team.total_team_pure / self.config.MAX_TEAM_STAT_VALUE
        team_obs[2] = team.total_team_cool / self.config.MAX_TEAM_STAT_VALUE

        # Next 9 features: SIS slot usage per team slot
        for i, slot in enumerate(team.slots):
            if slot.card_sis_capacity > 0:
                sis_usage_ratio = slot.total_sis_slots_used / slot.card_sis_capacity
            else:
                sis_usage_ratio = 0.0
            team_obs[3 + i] = sis_usage_ratio

        return team_obs

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
