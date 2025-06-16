import unittest
import os
import sys
from io import StringIO

from cardfactory import CardFactory
from deck import Deck

class TestDeck(unittest.TestCase):

    DECK_SAVE_PATH = './data/test_deck.json'

    def setUp(self):
        """Set up the factory, redirect stdout, and ensure clean file state for each test."""
        self.factory = CardFactory(cards_json_path="./data/cards.json", level_caps_json_path="./data/level_caps.json", level_cap_bonuses_path="./data/level_cap_bonuses.json")
        self.held_stdout = sys.stdout
        sys.stdout = self.captured_output = StringIO()

        if os.path.exists(self.DECK_SAVE_PATH):
            os.remove(self.DECK_SAVE_PATH)

    def tearDown(self):
        """Restore stdout and clean up test files."""
        sys.stdout = self.held_stdout

        if os.path.exists(self.DECK_SAVE_PATH):
            os.remove(self.DECK_SAVE_PATH)

    def test_add_card_display_deck(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(101, idolized=True, skill_level=4)
        test_deck.add_card(28, idolized=False)
        test_deck.add_card(96, idolized=True)
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 101 - Initial Nozomi (UR)
  State: Level: 100, Idolized: True, Skill Lvl: 4, SIS: 4
Deck ID: 2 | Card ID: 28 - Uniform / Natsuiro Egao de 1,2,Jump! Honoka (R)
  State: Level: 40, Idolized: False, Skill Lvl: 1, SIS: 1
Deck ID: 3 | Card ID: 96 - Season 1 BD Bonus Maki (UR)
  State: Level: 100, Idolized: True, Skill Lvl: 1, SIS: 2
Total cards in deck: 3
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_add_non_existent_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(4001, idolized=True, skill_level=4)
        test_deck.add_card(28, idolized=False)
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 28 - Uniform / Natsuiro Egao de 1,2,Jump! Honoka (R)
  State: Level: 40, Idolized: False, Skill Lvl: 1, SIS: 1
Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_modify_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000, idolized=False, skill_level=4)
        test_deck.modify_card(deck_id=1, idolized=True, skill_level=6)
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 1000 - Diving Mari (SR)
  State: Level: 80, Idolized: True, Skill Lvl: 6, SIS: 2
Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000, idolized=False, skill_level=4)
        test_deck.add_card(1001, idolized=False, skill_level=4)
        test_deck.add_card(1002, idolized=False, skill_level=4)
        test_deck.remove_card(2)
        test_deck.add_card(1000)
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 1000 - Diving Mari (SR)
  State: Level: 60, Idolized: False, Skill Lvl: 4, SIS: 2
Deck ID: 3 | Card ID: 1002 - Pool Kotori (UR)
  State: Level: 80, Idolized: False, Skill Lvl: 4, SIS: 4
Deck ID: 4 | Card ID: 1000 - Diving Mari (SR)
  State: Level: 60, Idolized: False, Skill Lvl: 1, SIS: 2
Total cards in deck: 3
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_invalid_deck_id(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000, idolized=False, skill_level=4)
        with self.assertWarns(UserWarning) as cm:
            test_deck.remove_card(2)
        self.assertEqual(str(cm.warnings[0].message), "Deck ID 2 not found for removal.")
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 1000 - Diving Mari (SR)
  State: Level: 60, Idolized: False, Skill Lvl: 4, SIS: 2
Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_save_deck(self):
        test_deck = Deck(self.factory)
        self.assertFalse(os.path.exists(self.DECK_SAVE_PATH))
        test_deck.add_card(1000, idolized=False, level=110, sis_slots=2)
        test_deck.save_deck(self.DECK_SAVE_PATH)
        self.assertTrue(os.path.exists(self.DECK_SAVE_PATH))

    def test_load_deck(self):
        # First, create and save a deck to load from
        deck_to_save = Deck(self.factory)
        deck_to_save.add_card(1000, idolized=False, level=110, sis_slots=2)
        deck_to_save.save_deck(self.DECK_SAVE_PATH)

        # Now, create a new deck, populate it, and then load over it
        test_deck = Deck(self.factory)
        test_deck.add_card(1500)
        test_deck.add_card(1600)
        test_deck.load_deck(self.DECK_SAVE_PATH)
        test_deck.display_deck()
        expected = """--- Current Deck Contents ---
Deck ID: 1 | Card ID: 1000 - Diving Mari (SR)
  State: Level: 60, Idolized: False, Skill Lvl: 1, SIS: 2
Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_delete_deck(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000)
        test_deck.add_card(1500)
        test_deck.delete_deck()
        test_deck.display_deck()
        self.assertEqual(self.captured_output.getvalue().strip(), "Deck is currently empty.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
