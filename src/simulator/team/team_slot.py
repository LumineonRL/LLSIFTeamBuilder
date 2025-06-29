import warnings
from typing import List, Optional

from src.simulator.card.card import Card
from src.simulator.card.deck import DeckEntry
from src.simulator.accessory.accessory import Accessory
from src.simulator.accessory.accessory_manager import PlayerAccessory
from src.simulator.sis.sis import SIS
from src.simulator.sis.sis_manager import PlayerSIS


class TeamSlot:
    """
    Represents a single position on a team.

    This class acts as a container that links a Card instance to its equipped
    Accessory and list of School Idol Skills (SIS). It also handles the
    validation for equipping SIS based on the card's available slots.
    """

    def __init__(self):
        self.card_entry: Optional[DeckEntry] = None
        self.accessory_entry: Optional[PlayerAccessory] = None
        self.sis_entries: List[PlayerSIS] = []

        self.total_smile: int = 0
        self.total_pure: int = 0
        self.total_cool: int = 0

    @property
    def card(self) -> Optional[Card]:
        """Convenience property to access the Card object directly."""
        return self.card_entry.card if self.card_entry else None

    @property
    def accessory(self) -> Optional[Accessory]:
        """Convenience property to access the Accessory object directly."""
        return self.accessory_entry.accessory if self.accessory_entry else None

    @property
    def sis_list(self) -> List[SIS]:
        """Convenience property to access the list of SIS objects directly."""
        return [ps.sis for ps in self.sis_entries]

    @property
    def total_sis_slots_used(self) -> int:
        """Calculates the total number of SIS slots currently used."""
        return sum(sis.sis.slots for sis in self.sis_entries)

    @property
    def card_sis_capacity(self) -> int:
        """Returns the total number of SIS slots the equipped card has."""
        if not self.card:
            return 0
        return self.card.current_sis_slots

    @property
    def available_sis_slots(self) -> int:
        """Calculates the remaining number of SIS slots available."""
        return self.card_sis_capacity - self.total_sis_slots_used

    def equip_card(self, card_entry: DeckEntry) -> None:
        """Equips a card to this slot, clearing any existing items."""
        self.clear()
        self.card_entry = card_entry

    def equip_accessory(self, accessory_entry: PlayerAccessory) -> bool:
        """Equips an accessory to this slot."""
        if not self.card:
            warnings.warn("Cannot equip an accessory: No card is in this slot.")
            return False
        self.accessory_entry = accessory_entry
        return True

    def equip_sis(self, sis_entry: PlayerSIS) -> bool:
        """
        Equips a SIS to this slot.
        Assumes all validation (capacity, restrictions) has been performed externally.
        """
        if not self.card:
            warnings.warn("Cannot equip SIS: No card is in this slot.")
            return False

        self.sis_entries.append(sis_entry)
        return True

    def unequip_sis(self, manager_internal_id: int) -> bool:
        """Unequips a SIS from the slot by its manager-specific internal ID."""
        initial_count = len(self.sis_entries)
        self.sis_entries = [
            ps
            for ps in self.sis_entries
            if ps.manager_internal_id != manager_internal_id
        ]
        return len(self.sis_entries) < initial_count

    def clear(self) -> None:
        """Clears the slot of all cards, accessories, and SIS."""
        self.card_entry = None
        self.accessory_entry = None
        self.sis_entries.clear()
        self.total_smile = 0
        self.total_pure = 0
        self.total_cool = 0

    def __repr__(self) -> str:
        """Provides a detailed, multi-line string representation of the slot."""
        if not self.card_entry or not self.card_entry.card:
            return "  <Empty Slot>"

        # Card line
        card_line = f"  Card: {self.card_entry.card.display_name} (Deck ID: {self.card_entry.deck_id})"
        stats_line = (
            f"  Stats: S/P/C {self.total_smile}/{self.total_pure}/{self.total_cool}"
        )

        # Accessory line
        if self.accessory_entry and self.accessory_entry.accessory:
            acc_line = f"  Accessory: {self.accessory_entry.accessory.name} (Manager ID: {self.accessory_entry.manager_internal_id})"
        else:
            acc_line = "  Accessory: None"

        # SIS lines
        sis_lines = []
        if self.sis_list:
            sis_header = f"  SIS ({self.total_sis_slots_used}/{self.card_sis_capacity} slots used):"
            sis_items = [
                f"    - {sis.name} ({sis.slots} slots)" for sis in self.sis_list
            ]
            sis_lines.extend([sis_header, *sis_items])
        else:
            sis_lines.append("  SIS: None")

        return "\n".join([card_line, stats_line, acc_line, *sis_lines])
