from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from src.simulator import Team
from src.team_builder.env.build_phase import BuildPhase
from .serializer import Serializer

if TYPE_CHECKING:
    from .config import EnvConfig
    from .env import LLSIFTeamBuildingEnv


class ObservationManager:
    """
    Manages observation space definition and observation creation by
    delegating serialization tasks to the Serializer class.
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
        self.serializer = Serializer(config)

    def define_action_space(self) -> spaces.Discrete:
        """
        Defines the action space for the environment based on the maximum
        possible items defined in the configuration.
        """
        max_actions = max(
            self.config.MAX_CARDS_IN_DECK,
            self.config.MAX_ACCESSORIES_IN_INVENTORY + self.ACTION_ID_OFFSET,
            self.config.MAX_SIS_IN_INVENTORY + self.ACTION_ID_OFFSET,
            self.config.MAX_GUESTS,
            self.APPROACH_RATE_CHOICES,
        )
        return spaces.Discrete(max_actions)

    def define_observation_space(self) -> spaces.Dict:
        """Defines the observation space for the environment."""
        c = self.config
        s = self.serializer
        attribute_size = len(c.ATTRIBUTE_MAP)
        char_map_size = len(c.character_map)
        skill_param_shape = (c.MAX_SKILL_LIST_ENTRIES, 4)
        notes_shape = (c.MAX_NOTE_COUNT, s.note_feature_size)

        return spaces.Dict(
            {
                "deck": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, s.card_feature_size), np.float32
                ),
                "team_cards": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, s.card_feature_size), np.float32
                ),
                "song": spaces.Box(0, 1, (s.song_feature_size,), np.float32),
                "notes": spaces.Box(0, 1, notes_shape, np.float32),
                "sis": spaces.Box(
                    0, 1, (c.MAX_SIS_IN_INVENTORY, s.sis_feature_size), np.float32
                ),
                "team_sis": spaces.Box(
                    0,
                    1,
                    (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, s.sis_feature_size),
                    np.float32,
                ),
                "accessories": spaces.Box(
                    0,
                    1,
                    (c.MAX_ACCESSORIES_IN_INVENTORY, s.accessory_feature_size),
                    np.float32,
                ),
                "guest": spaces.Box(
                    0, 1, (c.MAX_GUESTS, s.guest_feature_size), np.float32
                ),
                "team_accessories": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, s.accessory_feature_size), np.float32
                ),
                "team_guest": spaces.Box(0, 1, (1, s.guest_feature_size), np.float32),
                # Attribute spaces
                "deck_attribute": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, attribute_size), np.float32
                ),
                "team_cards_attribute": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, attribute_size), np.float32
                ),
                "song_attribute": spaces.Box(0, 1, (attribute_size,), np.float32),
                "sis_attribute": spaces.Box(
                    0, 1, (c.MAX_SIS_IN_INVENTORY, attribute_size), np.float32
                ),
                "team_sis_attribute": spaces.Box(
                    0,
                    1,
                    (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, attribute_size),
                    np.float32,
                ),
                "card_ls_attribute": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, attribute_size), np.float32
                ),
                "card_ls_secondary_attribute": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, attribute_size), np.float32
                ),
                "card_ls_extra_attribute": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, attribute_size), np.float32
                ),
                "team_card_ls_attribute": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, attribute_size), np.float32
                ),
                "team_card_ls_secondary_attribute": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, attribute_size), np.float32
                ),
                "team_card_ls_extra_attribute": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, attribute_size), np.float32
                ),
                "guest_ls_attribute": spaces.Box(
                    0, 1, (c.MAX_GUESTS, attribute_size), np.float32
                ),
                "guest_ls_secondary_attribute": spaces.Box(
                    0, 1, (c.MAX_GUESTS, attribute_size), np.float32
                ),
                "guest_ls_extra_attribute": spaces.Box(
                    0, 1, (c.MAX_GUESTS, attribute_size), np.float32
                ),
                "team_guest_ls_attribute": spaces.Box(
                    0, 1, (1, attribute_size), np.float32
                ),
                "team_guest_ls_secondary_attribute": spaces.Box(
                    0, 1, (1, attribute_size), np.float32
                ),
                "team_guest_ls_extra_attribute": spaces.Box(
                    0, 1, (1, attribute_size), np.float32
                ),
                "card_character": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, char_map_size), np.float32
                ),
                "card_ls_extra_target": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, char_map_size), np.float32
                ),
                "card_skill_target": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK, char_map_size), np.float32
                ),
                "team_card_character": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, char_map_size), np.float32
                ),
                "team_card_ls_extra_target": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, char_map_size), np.float32
                ),
                "team_card_skill_target": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, char_map_size), np.float32
                ),
                "acc_character": spaces.Box(
                    0, 1, (c.MAX_ACCESSORIES_IN_INVENTORY, char_map_size), np.float32
                ),
                "team_acc_character": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, char_map_size), np.float32
                ),
                "accessory_skill_target": spaces.Box(
                    0, 1, (c.MAX_ACCESSORIES_IN_INVENTORY, char_map_size), np.float32
                ),
                "team_accessory_skill_target": spaces.Box(
                    0, 1, (Team.NUM_SLOTS, char_map_size), np.float32
                ),
                "sis_group": spaces.Box(
                    0, 1, (c.MAX_SIS_IN_INVENTORY, char_map_size), np.float32
                ),
                "sis_equip_restriction": spaces.Box(
                    0, 1, (c.MAX_SIS_IN_INVENTORY, char_map_size), np.float32
                ),
                "team_sis_group": spaces.Box(
                    0,
                    1,
                    (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, char_map_size),
                    np.float32,
                ),
                "team_sis_equip_restriction": spaces.Box(
                    0,
                    1,
                    (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, char_map_size),
                    np.float32,
                ),
                "guest_extra_target": spaces.Box(
                    0, 1, (c.MAX_GUESTS, char_map_size), np.float32
                ),
                "team_guest_extra_target": spaces.Box(
                    0, 1, (1, char_map_size), np.float32
                ),
                "song_group": spaces.Box(0, 1, (char_map_size,), np.float32),
                "card_skill_params": spaces.Box(
                    0, 1, (c.MAX_CARDS_IN_DECK,) + skill_param_shape, np.float32
                ),
                "team_card_skill_params": spaces.Box(
                    0, 1, (Team.NUM_SLOTS,) + skill_param_shape, np.float32
                ),
                "accessory_skill_params": spaces.Box(
                    0,
                    1,
                    (c.MAX_ACCESSORIES_IN_INVENTORY,) + skill_param_shape,
                    np.float32,
                ),
                "team_accessory_skill_params": spaces.Box(
                    0, 1, (Team.NUM_SLOTS,) + skill_param_shape, np.float32
                ),
                "build_phase": spaces.Discrete(len(BuildPhase)),
                "current_slot": spaces.Discrete(Team.NUM_SLOTS),
                "accuracy": spaces.Box(0, 1, (1,), np.float32),
            }
        )

    def get_obs(self) -> Dict[str, Any]:
        """
        Constructs the current observation from the environment state.

        Returns:
            A dictionary containing the serialized state matching the
            flattened observation space.
        """
        c = self.config
        s = self.serializer
        char_map_size = len(c.character_map)
        attr_size = len(c.ATTRIBUTE_MAP)
        skill_param_shape = (c.MAX_SKILL_LIST_ENTRIES, 4)

        # --- Inventory Observations ---
        deck_obs = np.zeros((c.MAX_CARDS_IN_DECK, s.card_feature_size), np.float32)
        deck_attr = np.zeros((c.MAX_CARDS_IN_DECK, attr_size), np.float32)
        card_ls_attr = np.zeros((c.MAX_CARDS_IN_DECK, attr_size), np.float32)
        card_ls_sec_attr = np.zeros((c.MAX_CARDS_IN_DECK, attr_size), np.float32)
        card_ls_ext_attr = np.zeros((c.MAX_CARDS_IN_DECK, attr_size), np.float32)
        card_char = np.zeros((c.MAX_CARDS_IN_DECK, char_map_size), np.float32)
        card_ls_target = np.zeros((c.MAX_CARDS_IN_DECK, char_map_size), np.float32)
        card_skill_target = np.zeros((c.MAX_CARDS_IN_DECK, char_map_size), np.float32)
        card_skill_params = np.zeros(
            (c.MAX_CARDS_IN_DECK,) + skill_param_shape, np.float32
        )

        for i, entry in enumerate(
            sorted(self.env.deck.entries.values(), key=lambda x: x.deck_id)
        ):
            if i >= c.MAX_CARDS_IN_DECK:
                break
            card = entry.card
            ls = card.leader_skill
            deck_obs[i] = s.serialize_card(card)
            deck_attr[i] = s.serialize_attribute(card.attribute)
            card_ls_attr[i] = s.serialize_attribute(ls.attribute)
            card_ls_sec_attr[i] = s.serialize_attribute(ls.secondary_attribute)
            card_ls_ext_attr[i] = s.serialize_attribute(ls.extra_attribute)
            card_char[i] = s.one_hot_mapped(card.character, c.character_map)
            card_ls_target[i] = s.get_multi_hot_vector(
                c.ls_extra_target_map, ls.extra_target, c.ls_extra_target_default_vector
            )
            card_skill_target[i] = s.get_multi_hot_vector(
                c.skill_target_map, card.skill.target, c.skill_target_default_vector
            )
            card_skill_params[i] = s.serialize_skill_parameters(card)

        acc_obs = np.zeros(
            (c.MAX_ACCESSORIES_IN_INVENTORY, s.accessory_feature_size), np.float32
        )
        acc_char = np.zeros((c.MAX_ACCESSORIES_IN_INVENTORY, char_map_size), np.float32)
        acc_skill_target = np.zeros(
            (c.MAX_ACCESSORIES_IN_INVENTORY, char_map_size), np.float32
        )
        acc_skill_params = np.zeros(
            (c.MAX_ACCESSORIES_IN_INVENTORY,) + skill_param_shape, np.float32
        )

        for i, pa in enumerate(
            sorted(
                self.env.accessory_manager.accessories.values(),
                key=lambda x: x.manager_internal_id,
            )
        ):
            if i >= c.MAX_ACCESSORIES_IN_INVENTORY:
                break
            acc = pa.accessory
            acc_obs[i] = s.serialize_accessory(acc)
            acc_char[i] = s.one_hot_mapped(acc.character, c.character_map)
            acc_skill_target[i] = s.get_multi_hot_vector(
                c.skill_target_map, acc.skill.target, c.skill_target_default_vector
            )
            acc_skill_params[i] = s.serialize_skill_parameters(acc)

        sis_obs = np.zeros((c.MAX_SIS_IN_INVENTORY, s.sis_feature_size), np.float32)
        sis_attr = np.zeros((c.MAX_SIS_IN_INVENTORY, attr_size), np.float32)
        sis_group = np.zeros((c.MAX_SIS_IN_INVENTORY, char_map_size), np.float32)
        sis_equip = np.zeros((c.MAX_SIS_IN_INVENTORY, char_map_size), np.float32)

        for i, ps in enumerate(
            sorted(
                self.env.sis_manager.skills.values(),
                key=lambda x: x.manager_internal_id,
            )
        ):
            if i >= c.MAX_SIS_IN_INVENTORY:
                break
            sis = ps.sis
            sis_obs[i] = s.serialize_sis(sis)
            sis_attr[i] = s.serialize_attribute(sis.attribute)
            sis_group[i] = s.get_multi_hot_vector(
                c.sis_group_map, sis.group, c.sis_group_default_vector
            )
            sis_equip[i] = s.get_multi_hot_vector(
                c.sis_equip_restriction_map,
                sis.equip_restriction,
                c.sis_equip_restriction_default_vector,
            )

        guest_obs = np.zeros((c.MAX_GUESTS, s.guest_feature_size), np.float32)
        guest_ls_attr = np.zeros((c.MAX_GUESTS, attr_size), np.float32)
        guest_ls_sec_attr = np.zeros((c.MAX_GUESTS, attr_size), np.float32)
        guest_ls_ext_attr = np.zeros((c.MAX_GUESTS, attr_size), np.float32)
        guest_target = np.zeros((c.MAX_GUESTS, char_map_size), np.float32)
        if self.env.guest_manager:
            for i, guest in enumerate(
                sorted(
                    self.env.guest_manager.all_guests.values(),
                    key=lambda g: g.leader_skill_id,
                )
            ):
                if i >= c.MAX_GUESTS:
                    break
                guest_obs[i] = s.serialize_guest(guest)
                guest_ls_attr[i] = s.serialize_attribute(guest.leader_attribute)
                guest_ls_sec_attr[i] = s.serialize_attribute(
                    guest.leader_secondary_attribute
                )
                guest_ls_ext_attr[i] = s.serialize_attribute(
                    guest.leader_extra_attribute
                )
                guest_target[i] = s.get_multi_hot_vector(
                    c.ls_extra_target_map,
                    guest.leader_extra_target,
                    c.ls_extra_target_default_vector,
                )

        # --- Team, Song, and State Observations ---
        team_obs_dict = self._serialize_team(self.env.state.team)
        song_obs = s.serialize_song(self.env.song)
        song_notes_obs = s.serialize_notes(self.env.song)
        song_attribute_obs = s.serialize_attribute(
            self.env.song.attribute if self.env.song else None
        )
        song_group_obs = (
            s.get_multi_hot_vector(
                c.sis_group_map, self.env.song.group, c.sis_group_default_vector
            )
            if self.env.song
            else np.zeros(char_map_size, dtype=np.float32)
        )

        return {
            "deck": deck_obs,
            "deck_attribute": deck_attr,
            "card_character": card_char,
            "card_ls_extra_target": card_ls_target,
            "card_skill_target": card_skill_target,
            "card_skill_params": card_skill_params,
            "accessories": acc_obs,
            "acc_character": acc_char,
            "accessory_skill_target": acc_skill_target,
            "accessory_skill_params": acc_skill_params,
            "sis": sis_obs,
            "sis_attribute": sis_attr,
            "sis_group": sis_group,
            "sis_equip_restriction": sis_equip,
            "guest": guest_obs,
            "guest_extra_target": guest_target,
            "song": song_obs,
            "notes": song_notes_obs,
            "song_attribute": song_attribute_obs,
            "song_group": song_group_obs,
            "card_ls_attribute": card_ls_attr,
            "card_ls_secondary_attribute": card_ls_sec_attr,
            "card_ls_extra_attribute": card_ls_ext_attr,
            "guest_ls_attribute": guest_ls_attr,
            "guest_ls_secondary_attribute": guest_ls_sec_attr,
            "guest_ls_extra_attribute": guest_ls_ext_attr,
            **team_obs_dict,
            "build_phase": int(self.env.state.build_phase),
            "current_slot": self.env.state.current_slot_idx,
            "accuracy": np.array([self.env.accuracy], dtype=np.float32),
        }

    def _serialize_team(self, team: Optional[Team]) -> Dict[str, np.ndarray]:
        """Converts the Team object into a dictionary of observation arrays."""
        c, s = self.config, self.serializer
        char_map_size = len(c.character_map)
        attr_size = len(c.ATTRIBUTE_MAP)
        skill_param_shape = (c.MAX_SKILL_LIST_ENTRIES, 4)

        obs = {
            "team_cards": np.zeros((Team.NUM_SLOTS, s.card_feature_size), np.float32),
            "team_cards_attribute": np.zeros((Team.NUM_SLOTS, attr_size), np.float32),
            "team_card_ls_attribute": np.zeros((Team.NUM_SLOTS, attr_size), np.float32),
            "team_card_ls_secondary_attribute": np.zeros(
                (Team.NUM_SLOTS, attr_size), np.float32
            ),
            "team_card_ls_extra_attribute": np.zeros(
                (Team.NUM_SLOTS, attr_size), np.float32
            ),
            "team_accessories": np.zeros(
                (Team.NUM_SLOTS, s.accessory_feature_size), np.float32
            ),
            "team_sis": np.zeros(
                (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, s.sis_feature_size), np.float32
            ),
            "team_sis_attribute": np.zeros(
                (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, attr_size), np.float32
            ),
            "team_guest": np.zeros((1, s.guest_feature_size), np.float32),
            "team_guest_ls_attribute": np.zeros((1, attr_size), np.float32),
            "team_guest_ls_secondary_attribute": np.zeros((1, attr_size), np.float32),
            "team_guest_ls_extra_attribute": np.zeros((1, attr_size), np.float32),
            "team_card_character": np.zeros(
                (Team.NUM_SLOTS, char_map_size), np.float32
            ),
            "team_card_ls_extra_target": np.zeros(
                (Team.NUM_SLOTS, char_map_size), np.float32
            ),
            "team_card_skill_target": np.zeros(
                (Team.NUM_SLOTS, char_map_size), np.float32
            ),
            "team_acc_character": np.zeros((Team.NUM_SLOTS, char_map_size), np.float32),
            "team_accessory_skill_target": np.zeros(
                (Team.NUM_SLOTS, char_map_size), np.float32
            ),
            "team_sis_group": np.zeros(
                (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, char_map_size), np.float32
            ),
            "team_sis_equip_restriction": np.zeros(
                (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS, char_map_size), np.float32
            ),
            "team_guest_extra_target": np.zeros((1, char_map_size), np.float32),
            "team_card_skill_params": np.zeros(
                (Team.NUM_SLOTS,) + skill_param_shape, np.float32
            ),
            "team_accessory_skill_params": np.zeros(
                (Team.NUM_SLOTS,) + skill_param_shape, np.float32
            ),
        }

        if not team:
            return obs

        for i, slot in enumerate(team.slots):
            if slot.card:
                card = slot.card
                ls = card.leader_skill
                obs["team_cards"][i] = s.serialize_card(card)
                obs["team_cards_attribute"][i] = s.serialize_attribute(card.attribute)
                obs["team_card_ls_attribute"][i] = s.serialize_attribute(ls.attribute)
                obs["team_card_ls_secondary_attribute"][i] = s.serialize_attribute(
                    ls.secondary_attribute
                )
                obs["team_card_ls_extra_attribute"][i] = s.serialize_attribute(
                    ls.extra_attribute
                )
                obs["team_card_character"][i] = s.one_hot_mapped(
                    card.character, c.character_map
                )
                obs["team_card_ls_extra_target"][i] = s.get_multi_hot_vector(
                    c.ls_extra_target_map,
                    ls.extra_target,
                    c.ls_extra_target_default_vector,
                )
                obs["team_card_skill_target"][i] = s.get_multi_hot_vector(
                    c.skill_target_map, card.skill.target, c.skill_target_default_vector
                )
                obs["team_card_skill_params"][i] = s.serialize_skill_parameters(card)
            if slot.accessory:
                acc = slot.accessory
                obs["team_accessories"][i] = s.serialize_accessory(acc)
                obs["team_acc_character"][i] = s.one_hot_mapped(
                    acc.character, c.character_map
                )
                obs["team_accessory_skill_target"][i] = s.get_multi_hot_vector(
                    c.skill_target_map, acc.skill.target, c.skill_target_default_vector
                )
                obs["team_accessory_skill_params"][i] = s.serialize_skill_parameters(
                    acc
                )

        sis_idx = 0
        for slot in team.slots:
            for sis_entry in slot.sis_entries:
                if sis_idx >= (Team.NUM_SLOTS * c.MAX_EQUIPPABLE_SIS):
                    break
                sis = sis_entry.sis
                obs["team_sis"][sis_idx] = s.serialize_sis(sis)
                obs["team_sis_attribute"][sis_idx] = s.serialize_attribute(
                    sis.attribute
                )
                obs["team_sis_group"][sis_idx] = s.get_multi_hot_vector(
                    c.sis_group_map, sis.group, c.sis_group_default_vector
                )
                obs["team_sis_equip_restriction"][sis_idx] = s.get_multi_hot_vector(
                    c.sis_equip_restriction_map,
                    sis.equip_restriction,
                    c.sis_equip_restriction_default_vector,
                )
                sis_idx += 1

        if team.guest_manager and team.guest_manager.current_guest:
            guest = team.guest_manager.current_guest
            obs["team_guest"][0] = s.serialize_guest(guest)
            obs["team_guest_ls_attribute"][0] = s.serialize_attribute(
                guest.leader_attribute
            )
            obs["team_guest_ls_secondary_attribute"][0] = s.serialize_attribute(
                guest.leader_secondary_attribute
            )
            obs["team_guest_ls_extra_attribute"][0] = s.serialize_attribute(
                guest.leader_extra_attribute
            )
            obs["team_guest_extra_target"][0] = s.get_multi_hot_vector(
                c.ls_extra_target_map,
                guest.leader_extra_target,
                c.ls_extra_target_default_vector,
            )

        return obs
