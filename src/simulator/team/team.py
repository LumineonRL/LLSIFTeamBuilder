import os
import json
import warnings
import math
from typing import List, Optional, Set, Dict

from src.simulator.card.card import Card
from src.simulator.card.deck import Deck
from src.simulator.core.leader_skill import LeaderSkill
from src.simulator.accessory.accessory import Accessory
from src.simulator.accessory.accessory_manager import AccessoryManager
from src.simulator.sis.sis import SIS
from src.simulator.sis.sis_manager import SISManager
from src.simulator.team.guest import Guest
from src.simulator.team.team_slot import TeamSlot


class Team:
    """
    Manages the 9 slots of a team, orchestrating assignments and calculations.
    """

    NUM_SLOTS = 9
    CENTER_SLOT_NUMBER = 5

    def _load_json_mapping(
        self, filepath: str, warning_message: str
    ) -> Dict[str, Set[str]]:
        """Generic helper to load a JSON file mapping groups to character sets."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            return {group: set(characters) for group, characters in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            warnings.warn(f"{warning_message} from '{filepath}': {e}.")
            return {}

    def __init__(
        self,
        deck: Deck,
        accessory_manager: AccessoryManager,
        sis_manager: SISManager,
        guest_manager: Optional[Guest] = None,
    ):
        self.deck = deck
        self.accessory_manager = accessory_manager
        self.sis_manager = sis_manager
        self.guest_manager = guest_manager

        self.slots: List[TeamSlot] = [TeamSlot() for _ in range(self.NUM_SLOTS)]

        self.assigned_deck_ids: Set[int] = set()
        self.assigned_accessory_ids: Set[int] = set()
        self.assigned_sis_ids: Set[int] = set()

        self.total_team_smile: int = 0
        self.total_team_pure: int = 0
        self.total_team_cool: int = 0

        self._year_group_mapping = self._load_json_mapping(
            os.path.join("data", "year_group_mapping.json"),
            "Could not load year group mapping",
        )
        self._group_member_mapping = self._load_json_mapping(
            os.path.join("data", "group_member_map.json"),
            "Could not load group member mapping for Nonets",
        )
        self._additional_skill_map = self._load_json_mapping(
            os.path.join("data", "additional_leader_skill_map.json"),
            "Could not load additional leader skill map",
        )

    @property
    def center_slot(self) -> TeamSlot:
        """Returns the center slot of the team."""
        return self.slots[self.CENTER_SLOT_NUMBER - 1]

    def _get_slot(self, slot_number: int) -> Optional[TeamSlot]:
        """Internal helper to validate and retrieve a slot using 1-based indexing."""
        if not 1 <= slot_number <= self.NUM_SLOTS:
            warnings.warn(
                f"Invalid slot number: {slot_number}. Must be between 1 and {self.NUM_SLOTS}."
            )
            return None
        return self.slots[slot_number - 1]

    # --- Equipment Methods ---

    def equip_card_in_slot(self, slot_number: int, deck_id: int) -> bool:
        """Equips a card from the deck to a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)
        if not slot:
            return False

        if deck_id in self.assigned_deck_ids:
            warnings.warn(
                f"Card with Deck ID {deck_id} is already assigned to another slot."
            )
            return False

        card_entry = self.deck.get_entry(deck_id)
        if not card_entry:
            warnings.warn(f"Card with Deck ID {deck_id} not found in the deck.")
            return False

        # If a card is already in the slot, clear the slot first
        self.clear_slot(slot_number)

        slot.equip_card(card_entry)
        self.assigned_deck_ids.add(deck_id)
        self.calculate_team_stats()
        return True

    def _check_accessory_id_restriction(self, card: Card, accessory: Accessory) -> bool:
        """
        Checks if an accessory can be equipped to a card based on card_id.
        Returns True if allowed, False otherwise.
        """
        card_id_to_match = card.card_id
        accessory_card_id_str = accessory.card_id

        # If the accessory has no specific card_id restriction, it can be equipped.
        if not accessory_card_id_str:
            return True

        try:
            accessory_card_id_int = int(accessory_card_id_str)
            if accessory_card_id_int == card_id_to_match:
                return True
        except (ValueError, TypeError):
            pass

        # If we reach here, the check failed.
        warnings.warn(
            f"Cannot equip accessory '{accessory.name}': "
            f"ID mismatch. Accessory requires card ID '{accessory.card_id}', "
            f"but card's ID is '{card.card_id}'."
        )
        return False

    def equip_accessory_in_slot(
        self, slot_number: int, manager_internal_id: int
    ) -> bool:
        """Equips an accessory to a card in a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)

        if not slot or not slot.card:
            warnings.warn(f"Cannot equip accessory: No card in slot {slot_number}.")
            return False

        if manager_internal_id in self.assigned_accessory_ids and (
            not slot.accessory_entry
            or slot.accessory_entry.manager_internal_id != manager_internal_id
        ):
            warnings.warn(
                f"Accessory with Manager ID {manager_internal_id} is already assigned."
            )
            return False

        acc_entry = self.accessory_manager.get_player_accessory(manager_internal_id)
        if not acc_entry:
            warnings.warn(f"Accessory with Manager ID {manager_internal_id} not found.")
            return False

        if not self._check_accessory_id_restriction(slot.card, acc_entry.accessory):
            return False

        # Unequip any existing accessory in this slot
        if slot.accessory_entry:
            self.assigned_accessory_ids.discard(
                slot.accessory_entry.manager_internal_id
            )

        if slot.equip_accessory(acc_entry):
            self.assigned_accessory_ids.add(manager_internal_id)
            self.calculate_team_stats()
            return True
        return False

    def _check_sis_attribute_restriction(self, card: Card, sis: SIS) -> bool:
        """Checks if a SIS attribute restriction is met."""
        if sis.equip_restriction == card.attribute:
            return True
        warnings.warn(
            f"Cannot equip SIS '{sis.name}': Attribute mismatch. "
            f"SIS requires {sis.equip_restriction}, but card is {card.attribute}."
        )
        return False

    def _check_sis_year_group_restriction(self, card: Card, sis: SIS) -> bool:
        """Checks if a SIS year group restriction is met."""
        restriction = sis.equip_restriction
        if not restriction:
            return False

        if card.character in self._year_group_mapping.get(restriction, set()):
            return True

        warnings.warn(
            f"Cannot equip SIS '{sis.name}': Year group mismatch. "
            f"SIS is for {restriction}, but '{card.character}' is not."
        )
        return False

    def _check_sis_character_restriction(self, card: Card, sis: SIS) -> bool:
        """Checks if a SIS character restriction is met."""
        if sis.equip_restriction == card.character:
            return True
        warnings.warn(
            f"Cannot equip SIS '{sis.name}': Character mismatch. "
            f"SIS requires '{sis.equip_restriction}', but card is '{card.character}'."
        )
        return False

    def _check_sis_equip_restriction(self, card: Card, sis: SIS) -> bool:
        """
        Checks if a SIS can be equipped to a card by dispatching to the correct validator.
        Returns True if allowed, False otherwise.
        """
        restriction = sis.equip_restriction
        if not restriction:  # Covers None and ""
            return True

        if restriction in ["Smile", "Pure", "Cool"]:
            return self._check_sis_attribute_restriction(card, sis)

        if restriction in self._year_group_mapping:
            return self._check_sis_year_group_restriction(card, sis)

        # Default case is to assume a character-specific restriction
        return self._check_sis_character_restriction(card, sis)

    def equip_sis_in_slot(self, slot_number: int, manager_internal_id: int) -> bool:
        """Equips a SIS to a card in a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)

        if not slot or not slot.card:
            warnings.warn(f"Cannot equip SIS: No card in slot {slot_number}.")
            return False

        if manager_internal_id in self.assigned_sis_ids:
            warnings.warn(
                f"SIS with Manager ID {manager_internal_id} is already assigned."
            )
            return False

        sis_entry = self.sis_manager.get_player_sis(manager_internal_id)
        if not sis_entry:
            warnings.warn(f"SIS with Manager ID {manager_internal_id} not found.")
            return False

        new_sis_id = sis_entry.sis.id
        if any(entry.sis.id == new_sis_id for entry in slot.sis_entries):
            warnings.warn(
                f"Cannot equip SIS '{sis_entry.sis.name}': "
                f"A SIS with the same ID ({new_sis_id}) is already in this slot."
            )
            return False

        if not self._check_sis_equip_restriction(slot.card, sis_entry.sis):
            return False

        if slot.equip_sis(sis_entry):
            self.assigned_sis_ids.add(manager_internal_id)
            self.calculate_team_stats()
            return True
        return False

    # --- Unequipment Methods ---

    def clear_slot(self, slot_number: int) -> bool:
        """Completely clears a slot of its card and any equipped items (1-9)."""
        slot = self._get_slot(slot_number)
        if not slot:
            return False

        for sis_entry in slot.sis_entries:
            self.assigned_sis_ids.discard(sis_entry.manager_internal_id)

        if slot.accessory_entry:
            self.assigned_accessory_ids.discard(
                slot.accessory_entry.manager_internal_id
            )

        if slot.card_entry:
            self.assigned_deck_ids.discard(slot.card_entry.deck_id)

        slot.clear()
        self.calculate_team_stats()
        return True

    # --- Calculations ---

    def _is_nonet_active(self, group_name: str) -> bool:
        """Checks if the team composition satisfies the requirements for a specific nonet group."""
        team_characters = {slot.card.character for slot in self.slots if slot.card}

        if len(team_characters) != 9:
            return False

        valid_members = self._group_member_mapping.get(group_name)
        if not valid_members:
            return False

        return team_characters.issubset(valid_members)

    def _calculate_all_percent_boosts(self) -> Dict[str, float]:
        """Calculates the total percentage boosts from all active 'all percent boost' SIS."""
        boosts = {"Smile": 0.0, "Pure": 0.0, "Cool": 0.0}
        active_nonets: Dict[str, bool] = {}

        for slot in self.slots:
            if not slot.card:
                continue
            for sis_entry in slot.sis_entries:
                sis = sis_entry.sis
                if sis.effect == "all percent boost":
                    if not sis.group:  # Regular boost (Aura, Veil)
                        if sis.attribute in boosts:
                            boosts[sis.attribute] += sis.value
                    else:  # Nonet boost
                        group_name = sis.group
                        if group_name not in active_nonets:
                            active_nonets[group_name] = self._is_nonet_active(
                                group_name
                            )

                        if active_nonets[group_name] and sis.attribute in boosts:
                            boosts[sis.attribute] += sis.value
        return boosts

    def _calculate_leader_skill_bonus(
        self, leader_skill: Optional[LeaderSkill], target_slot: TeamSlot
    ) -> Dict[str, int]:
        """Generic helper to calculate bonuses from a LeaderSkill object."""
        bonuses = {"Smile": 0, "Pure": 0, "Cool": 0}
        if not leader_skill or not target_slot.card:
            return bonuses

        primary_attr = leader_skill.attribute
        secondary_attr = leader_skill.secondary_attribute
        value = leader_skill.value

        if not secondary_attr:
            # On-attribute boost: affects the primary attribute based on its own value
            if primary_attr == "Smile":
                bonuses["Smile"] = math.ceil(target_slot.total_smile * value)
            elif primary_attr == "Pure":
                bonuses["Pure"] = math.ceil(target_slot.total_pure * value)
            elif primary_attr == "Cool":
                bonuses["Cool"] = math.ceil(target_slot.total_cool * value)
        else:
            # Off-attribute boost: affects the primary attribute based on the secondary attribute's value
            source_stat_map = {
                "Smile": target_slot.total_smile,
                "Pure": target_slot.total_pure,
                "Cool": target_slot.total_cool,
            }
            source_stat = source_stat_map.get(secondary_attr, 0)
            bonus_value = math.ceil(source_stat * value)
            if primary_attr in bonuses:
                bonuses[primary_attr] = bonus_value

        return bonuses

    def _calculate_extra_skill_bonus(
        self, extra_attr, extra_target, extra_value, target_slot
    ) -> Dict[str, int]:
        """Generic helper to calculate bonuses from an 'extra' skill component."""
        bonuses = {"Smile": 0, "Pure": 0, "Cool": 0}
        if not all([extra_attr, extra_target, extra_value, target_slot.card]):
            return bonuses

        target_group = self._additional_skill_map.get(extra_target, set())
        if target_slot.card.character in target_group:
            if extra_attr == "Smile":
                bonuses["Smile"] = math.ceil(target_slot.total_smile * extra_value)
            elif extra_attr == "Pure":
                bonuses["Pure"] = math.ceil(target_slot.total_pure * extra_value)
            elif extra_attr == "Cool":
                bonuses["Cool"] = math.ceil(target_slot.total_cool * extra_value)

        return bonuses

    def calculate_team_stats(self) -> None:
        """
        Calculates the stats for each slot and the team as a whole.
        """
        all_percent_boosts = self._calculate_all_percent_boosts()

        center_leader_skill = (
            self.center_slot.card.leader_skill if self.center_slot.card else None
        )
        guest = self.guest_manager.current_guest if self.guest_manager else None
        guest_leader_skill = (
            self.guest_manager.leader_skill if self.guest_manager else None
        )

        for slot in self.slots:
            # Step 1: Apply base card stats
            if not slot.card:
                slot.total_smile, slot.total_pure, slot.total_cool = 0, 0, 0
                continue

            slot.total_smile = slot.card.stats.smile
            slot.total_pure = slot.card.stats.pure
            slot.total_cool = slot.card.stats.cool

            # Step 2: Add Accessory Stats
            if slot.accessory:
                slot.total_smile += slot.accessory.stats.smile
                slot.total_pure += slot.accessory.stats.pure
                slot.total_cool += slot.accessory.stats.cool

            # Step 3: Apply team-wide "all percent boost" SIS (aura, veil, nonet, etc)
            slot.total_smile = math.ceil(
                slot.total_smile * (1 + all_percent_boosts.get("Smile", 0))
            )
            slot.total_pure = math.ceil(
                slot.total_pure * (1 + all_percent_boosts.get("Pure", 0))
            )
            slot.total_cool = math.ceil(
                slot.total_cool * (1 + all_percent_boosts.get("Cool", 0))
            )

            # Step 4: Apply slot-specific "self percent boost" SIS (ring, cross, etc)
            self_boosts = {"Smile": 0.0, "Pure": 0.0, "Cool": 0.0}
            for sis_entry in slot.sis_entries:
                sis = sis_entry.sis
                if sis.effect == "self percent boost" and sis.attribute in self_boosts:
                    self_boosts[sis.attribute] += sis.value

            slot.total_smile = math.ceil(slot.total_smile * (1 + self_boosts["Smile"]))
            slot.total_pure = math.ceil(slot.total_pure * (1 + self_boosts["Pure"]))
            slot.total_cool = math.ceil(slot.total_cool * (1 + self_boosts["Cool"]))

            # Step 5: Apply slot-specific "self flat boost" SIS (kiss, perfume, etc)
            for sis_entry in slot.sis_entries:
                sis = sis_entry.sis
                if sis.effect == "self flat boost":
                    if sis.attribute == "Smile":
                        slot.total_smile += int(sis.value)
                    elif sis.attribute == "Pure":
                        slot.total_pure += int(sis.value)
                    elif sis.attribute == "Cool":
                        slot.total_cool += int(sis.value)

            # Step 6: Apply Leader Skill and Guest Bonuses
            center_leader_bonus = self._calculate_leader_skill_bonus(
                center_leader_skill, slot
            )
            guest_leader_bonus = self._calculate_leader_skill_bonus(
                guest_leader_skill, slot
            )

            center_extra_bonus = (
                self._calculate_extra_skill_bonus(
                    center_leader_skill.extra_attribute,
                    center_leader_skill.extra_target,
                    center_leader_skill.extra_value,
                    slot,
                )
                if center_leader_skill
                else {"Smile": 0, "Pure": 0, "Cool": 0}
            )

            guest_extra_bonus = (
                self._calculate_extra_skill_bonus(
                    guest.leader_extra_attribute,
                    guest.leader_extra_target,
                    guest.leader_extra_value,
                    slot,
                )
                if guest
                else {"Smile": 0, "Pure": 0, "Cool": 0}
            )

            slot.total_smile += (
                center_leader_bonus["Smile"]
                + center_extra_bonus["Smile"]
                + guest_leader_bonus["Smile"]
                + guest_extra_bonus["Smile"]
            )
            slot.total_pure += (
                center_leader_bonus["Pure"]
                + center_extra_bonus["Pure"]
                + guest_leader_bonus["Pure"]
                + guest_extra_bonus["Pure"]
            )
            slot.total_cool += (
                center_leader_bonus["Cool"]
                + center_extra_bonus["Cool"]
                + guest_leader_bonus["Cool"]
                + guest_extra_bonus["Cool"]
            )

        self.total_team_smile = sum(s.total_smile for s in self.slots)
        self.total_team_pure = sum(s.total_pure for s in self.slots)
        self.total_team_cool = sum(s.total_cool for s in self.slots)

    def __repr__(self) -> str:
        """Provides a string representation of the entire team."""
        header = "--- Team Configuration ---"

        guest_line = "Guest: None"
        if self.guest_manager and self.guest_manager.current_guest:
            guest_line = f"Guest: {self.guest_manager.current_guest}"

        stats_header = (
            f"Total Stats: S/P/C "
            f"{self.total_team_smile}/{self.total_team_pure}/{self.total_team_cool}"
        )

        slot_details = []
        for i, slot in enumerate(self.slots):
            if slot.card:
                slot_header = f"\n[ Slot {i + 1} ]"
                slot_repr = repr(slot)
                slot_details.append(f"{slot_header}\n{slot_repr}")

        if not slot_details:
            return f"{header}\n{guest_line}\n{stats_header}\n<Team is empty>"

        footer = "--------------------------"

        return (
            f"{header}\n{guest_line}\n{stats_header}\n"
            + "\n".join(slot_details)
            + f"\n{footer}"
        )
