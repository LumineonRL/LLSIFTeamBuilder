import warnings
from typing import List, Optional, Set, Dict, Any

from card import Card
from deck import Deck
from accessory import Accessory
from accessorymanager import AccessoryManager
from sismanager import SISManager
from guest import Guest
from teamslot import TeamSlot

class Team:
    """
    Manages the 9 slots of a team, orchestrating assignments and calculations.
    """
    NUM_SLOTS = 9
    CENTER_SLOT_NUMBER = 5

    def __init__(self, deck: Deck, accessory_manager: AccessoryManager, sis_manager: SISManager, guest_manager: Optional[Guest] = None):
        self.deck = deck
        self.accessory_manager = accessory_manager
        self.sis_manager = sis_manager
        self.guest_manager = guest_manager

        self.slots: List[TeamSlot] = [TeamSlot() for _ in range(self.NUM_SLOTS)]

        self.assigned_deck_ids: Set[int] = set()
        self.assigned_accessory_ids: Set[int] = set()
        self.assigned_sis_ids: Set[int] = set()

    @property
    def center_slot(self) -> TeamSlot:
        """Returns the center slot of the team."""
        return self.slots[self.CENTER_SLOT_NUMBER - 1]

    def _get_slot(self, slot_number: int) -> Optional[TeamSlot]:
        """Internal helper to validate and retrieve a slot using 1-based indexing."""
        if not 1 <= slot_number <= self.NUM_SLOTS:
            warnings.warn(f"Invalid slot number: {slot_number}. Must be between 1 and {self.NUM_SLOTS}.")
            return None
        return self.slots[slot_number - 1]

    # --- Equipment Methods ---

    def equip_card_in_slot(self, slot_number: int, deck_id: int) -> bool:
        """Equips a card from the deck to a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)
        if not slot:
            return False

        if deck_id in self.assigned_deck_ids:
            warnings.warn(f"Card with Deck ID {deck_id} is already assigned to another slot.")
            return False

        card_entry = self.deck.get_entry(deck_id)
        if not card_entry:
            warnings.warn(f"Card with Deck ID {deck_id} not found in the deck.")
            return False

        # If a card is already in the slot, clear the slot first
        self.clear_slot(slot_number)

        slot.equip_card(card_entry)
        self.assigned_deck_ids.add(deck_id)
        return True

    def _check_accessory_character_restriction(self, card: Card, accessory: Accessory) -> bool:
        """
        Checks if an accessory can be equipped to a card based on character.
        Returns True if allowed, False otherwise.
        """
        card_character = card.character
        accessory_character = accessory.character

        # An empty string "" for accessory character means no restriction.
        if accessory_character == "" or accessory_character == card_character:
            return True

        warnings.warn(f"Cannot equip accessory '{accessory.name}': "
                      f"Character mismatch. Accessory is for '{accessory_character}', "
                      f"but card is for '{card_character}'.")
        return False

    def equip_accessory_in_slot(self, slot_number: int, manager_internal_id: int) -> bool:
        """Equips an accessory to a card in a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)

        if not slot or not slot.card:
            warnings.warn(f"Cannot equip accessory: No card in slot {slot_number}.")
            return False

        if manager_internal_id in self.assigned_accessory_ids and \
           (not slot.accessory_entry or slot.accessory_entry.manager_internal_id != manager_internal_id):
            warnings.warn(f"Accessory with Manager ID {manager_internal_id} is already assigned.")
            return False

        acc_entry = self.accessory_manager.get_player_accessory(manager_internal_id)
        if not acc_entry:
            warnings.warn(f"Accessory with Manager ID {manager_internal_id} not found.")
            return False

        if not self._check_accessory_character_restriction(slot.card, acc_entry.accessory):
            return False

        # Unequip any existing accessory in this slot
        if slot.accessory_entry:
            self.assigned_accessory_ids.discard(slot.accessory_entry.manager_internal_id)

        if slot.equip_accessory(acc_entry):
            self.assigned_accessory_ids.add(manager_internal_id)
            return True
        return False

    def equip_sis_in_slot(self, slot_number: int, manager_internal_id: int) -> bool:
        """Equips a SIS to a card in a specific team slot (1-9)."""
        slot = self._get_slot(slot_number)
        if not slot or not slot.card:
            warnings.warn(f"Cannot equip SIS: No card in slot {slot_number}.")
            return False

        if manager_internal_id in self.assigned_sis_ids:
            warnings.warn(f"SIS with Manager ID {manager_internal_id} is already assigned.")
            return False

        sis_entry = self.sis_manager.get_player_sis(manager_internal_id)
        if not sis_entry:
            warnings.warn(f"SIS with Manager ID {manager_internal_id} not found.")
            return False

        if slot.equip_sis(sis_entry):
            self.assigned_sis_ids.add(manager_internal_id)
            return True
        return False

    # --- Unequipment Methods ---

    def clear_slot(self, slot_number: int) -> bool:
        """Completely clears a slot of its card and any equipped items (1-9)."""
        slot = self._get_slot(slot_number)
        if not slot:
            return False

        # Unassign SIS
        for sis_entry in slot.sis_entries:
            self.assigned_sis_ids.discard(sis_entry.manager_internal_id)

        # Unassign Accessory
        if slot.accessory_entry:
            self.assigned_accessory_ids.discard(slot.accessory_entry.manager_internal_id)

        # Unassign Card
        if slot.card_entry:
            self.assigned_deck_ids.discard(slot.card_entry.deck_id)

        slot.clear()
        return True

    # --- Calculation Placeholders ---

    def calculate_team_stats(self) -> Dict[str, Any]:
        """
        (Placeholder) The main orchestrator for all stat calculations.
        """
        warnings.warn("Stat calculation logic is not yet implemented.")
        return {"status": "unimplemented"}

    def __repr__(self) -> str:
        """Provides a string representation of the entire team."""
        header = "--- Team Configuration ---"

        guest_line = "Guest: None"
        if self.guest_manager and self.guest_manager.current_guest:
            guest_line = f"Guest: {self.guest_manager.current_guest}"

        slot_details = []
        for i, slot in enumerate(self.slots):
            if slot.card:
                slot_header = f"\n[ Slot {i + 1} ]"
                slot_repr = repr(slot)
                slot_details.append(f"{slot_header}\n{slot_repr}")

        if not slot_details:
            return f"{header}\n{guest_line}\n<Team is empty>"

        footer = "--------------------------"

        return f"{header}\n{guest_line}\n" + "\n".join(slot_details) + f"\n{footer}"
