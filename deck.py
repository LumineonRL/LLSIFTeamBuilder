import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from card import Card
from cardfactory import CardFactory

@dataclass
class DeckEntry:
    """Container for a Card instance within a Deck."""
    deck_id: int
    card: Card

class Deck:
    """Manages a collection of Card instances and provides persistence."""

    def __init__(self, card_factory: CardFactory):
        self._card_factory = card_factory
        self._entries: Dict[int, DeckEntry] = {}
        self._next_deck_id: int = 1

    def add_card(self, card_id: int, **kwargs: Any) -> Optional[int]:
        """
        Creates a new card using the factory, assigns it a unique deck ID,
        and adds it to the deck.
        
        Args:
            card_id: The static ID of the card to add.
            **kwargs: Configuration for the card (e.g., idolized, skill_level).
        
        Returns:
            The unique deck_id of the newly added card, or None if creation failed.
        """
        card = self._card_factory.create_card(card_id, **kwargs)
        if not card:
            return None

        deck_id = self._next_deck_id
        self._entries[deck_id] = DeckEntry(deck_id=deck_id, card=card)
        self._next_deck_id += 1
        return deck_id

    def get_card(self, deck_id: int) -> Optional[Card]:
        """Retrieves a card instance from the deck by its unique deck_id."""
        entry = self._entries.get(deck_id)
        return entry.card if entry else None

    def remove_card(self, deck_id: int) -> bool:
        """Removes a card from the deck by its unique deck_id."""
        if deck_id in self._entries:
            del self._entries[deck_id]
            return True
        warnings.warn(f"Deck ID {deck_id} not found for removal.")
        return False

    def modify_card(self, deck_id: int, **kwargs: Any) -> bool:
        """
        Re-creates a card in the deck with modified attributes.
        This ensures the card's state is fully consistent after changes.
        """
        entry = self._entries.get(deck_id)
        if not entry:
            warnings.warn(f"Deck ID {deck_id} not found for modification.")
            return False

        current_card = entry.card

        current_config = {
            "idolized": current_card.idolized_status == 'idolized',
            "level": current_card.level,
            "skill_level": current_card.current_skill_level,
            "sis_slots": current_card.current_sis_slots
        }

        current_config.update(kwargs)

        new_card = self._card_factory.create_card(
            card_id=current_card.card_id,
            **current_config
        )

        if not new_card:
            warnings.warn(f"Failed to modify card with Deck ID {deck_id}. Re-creation failed.")
            return False

        entry.card = new_card
        return True

    def display_deck(self) -> None:
        """Prints a summary of each card currently in the deck."""
        if not self._entries:
            print("Deck is currently empty.")
            return

        print("\n--- Current Deck Contents ---")
        for deck_id, entry in self._entries.items():
            card = entry.card
            print(f"Deck ID: {deck_id} | Card ID: {card.card_id} - {card.display_name} ({card.rarity})")
            print(f"  State: Level: {card.level}, Idolized: {card.idolized_status == 'idolized'}, Skill Lvl: {card.current_skill_level}, SIS: {card.current_sis_slots}")
        print(f"Total cards in deck: {len(self._entries)}")
        print("--------------------------")

    def get_unassigned_cards(self, assigned_deck_ids: set[int]) -> List[Card]:
        """Returns a list of Card objects not in the provided set of assigned IDs."""
        return [entry.card for deck_id, entry in self._entries.items() if deck_id not in assigned_deck_ids]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the deck to a dictionary for JSON conversion."""
        return {
            "next_deck_id": self._next_deck_id,
            "entries": [
                {
                    "deck_id": entry.deck_id,
                    "card_id": entry.card.card_id,
                    "config": {
                        "idolized": entry.card.idolized_status == 'idolized',
                        "level": entry.card.level,
                        "skill_level": entry.card.current_skill_level,
                        "sis_slots": entry.card.current_sis_slots,
                    }
                } for entry in self._entries.values()
            ]
        }

    def save_deck(self, filepath: str) -> bool:
        """Saves the current deck state to a JSON file."""
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except (IOError, TypeError) as e:
            warnings.warn(f"Could not save deck to {filepath}: {e}")
            return False

    def load_deck(self, filepath: str) -> bool:
        """Loads a deck state from a JSON file, overwriting the current deck."""
        if not os.path.exists(filepath):
            warnings.warn(f"File not found at {filepath}. Cannot load deck.")
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            new_entries = {}
            for entry_data in data.get("entries", []):
                card_id = entry_data.get("card_id")
                config = entry_data.get("config", {})
                card = self._card_factory.create_card(card_id, **config)
                if card:
                    deck_id = entry_data["deck_id"]
                    new_entries[deck_id] = DeckEntry(deck_id=deck_id, card=card)
                else:
                    warnings.warn(f"Skipping card in deck file due to creation failure: card_id {card_id}")
            
            self._entries = new_entries
            self._next_deck_id = data.get("next_deck_id", 1)
            return True
        except (IOError, json.JSONDecodeError, KeyError) as e:
            warnings.warn(f"Could not load or parse deck file {filepath}: {e}")
            self.delete_deck()
            return False

    def delete_deck(self) -> None:
        """Clears all cards from the deck and resets the ID counter."""
        self._entries.clear()
        self._next_deck_id = 1
