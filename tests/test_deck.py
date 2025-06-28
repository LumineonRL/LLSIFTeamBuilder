import unittest
import os
import sys
from io import StringIO

from cardfactory import CardFactory
from deck import Deck
from gallery import Gallery

class TestDeck(unittest.TestCase):

    DECK_SAVE_PATH = './data/test_deck.json'

    def setUp(self):
        """Set up the factory, redirect stdout, and ensure clean file state for each test."""
        self.factory = CardFactory(cards_json_path="./data/cards.json", level_caps_json_path="./data/level_caps.json", level_cap_bonuses_path="./data/level_cap_bonuses.json")
        self.gallery = Gallery(0, 0, 0)

        self.held_stdout = sys.stdout
        sys.stdout = self.captured_output = StringIO()

        if os.path.exists(self.DECK_SAVE_PATH):
            os.remove(self.DECK_SAVE_PATH)

    def tearDown(self):
        """Restore stdout and clean up test files."""
        sys.stdout = self.held_stdout

        if os.path.exists(self.DECK_SAVE_PATH):
            os.remove(self.DECK_SAVE_PATH)

    def test_add_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(101, idolized=True, skill_level=4)
        test_deck.add_card(28, idolized=False)
        test_deck.add_card(96, idolized=True)
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=101 name='Initial Nozomi' rarity='UR'>
  - Info: Character='Tojo Nozomi', Attribute='Pure', Level=100, Idolized=True
  - Stats (S/P/C): 4190/6300/4300
  - Skill: Level=4, Type='Scorer'
    - Details: Activation: 'Combo'
    - Effects: Chance: 0.48%, Threshold: 17.0, Value: 1350.0
  - SIS Slots: 4 (Base: 4, Max: 8)
  - Leader Skill:
    - Main: Boosts 'Pure' by 9.0%
    - Extra: Boosts 'Pure' for 'μ's' by 3.0%

Deck ID: 2
<Card id=28 name='Uniform / Natsuiro Egao de 1,2,Jump! Honoka' rarity='R'>
  - Info: Character='Kosaka Honoka', Attribute='Smile', Level=40, Idolized=False
  - Stats (S/P/C): 3850/1590/1720
  - Skill: Level=1, Type='Scorer'
    - Details: Activation: 'Combo'
    - Effects: Chance: 0.36%, Threshold: 17.0, Value: 200.0
  - SIS Slots: 1 (Base: 1, Max: 1)
  - Leader Skill:
    - Main: Boosts 'Smile' by 3.0%

Deck ID: 3
<Card id=96 name='Season 1 BD Bonus Maki' rarity='UR'>
  - Info: Character='Nishikino Maki', Attribute='Cool', Level=100, Idolized=True
  - Stats (S/P/C): 3910/4060/5870
  - Skill: Level=1, Type='Scorer'
    - Details: Activation: 'Perfects'
    - Effects: Chance: 0.36%, Threshold: 15.0, Value: 200.0
  - SIS Slots: 2 (Base: 2, Max: 2)
  - Leader Skill:
    - Main: Boosts 'Cool' by 3.0%

Total cards in deck: 3
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_add_non_existent_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(4001, idolized=True, skill_level=4)
        test_deck.add_card(28, idolized=False)
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=28 name='Uniform / Natsuiro Egao de 1,2,Jump! Honoka' rarity='R'>
  - Info: Character='Kosaka Honoka', Attribute='Smile', Level=40, Idolized=False
  - Stats (S/P/C): 3850/1590/1720
  - Skill: Level=1, Type='Scorer'
    - Details: Activation: 'Combo'
    - Effects: Chance: 0.36%, Threshold: 17.0, Value: 200.0
  - SIS Slots: 1 (Base: 1, Max: 1)
  - Leader Skill:
    - Main: Boosts 'Smile' by 3.0%

Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_modify_card(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000, idolized=False, skill_level=4)
        test_deck.modify_card(deck_id=1, idolized=True, skill_level=6)
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=1000 name='Diving Mari' rarity='SR'>
  - Info: Character='Ohara Mari', Attribute='Smile', Level=80, Idolized=True
  - Stats (S/P/C): 5310/3410/3960
  - Skill: Level=6, Type='Perfect Lock'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.51%, Threshold: 26.0, Duration: 6.0s
  - SIS Slots: 2 (Base: 2, Max: 4)
  - Leader Skill:
    - Main: Boosts 'Smile' by 6.0%

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
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=1000 name='Diving Mari' rarity='SR'>
  - Info: Character='Ohara Mari', Attribute='Smile', Level=60, Idolized=False
  - Stats (S/P/C): 4780/3130/3680
  - Skill: Level=4, Type='Perfect Lock'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.45%, Threshold: 26.0, Duration: 5.0s
  - SIS Slots: 2 (Base: 2, Max: 2)
  - Leader Skill:
    - Main: Boosts 'Smile' by 6.0%

Deck ID: 3
<Card id=1002 name='Pool Kotori' rarity='UR'>
  - Info: Character='Minami Kotori', Attribute='Pure', Level=80, Idolized=False
  - Stats (S/P/C): 3820/5570/4120
  - Skill: Level=4, Type='Healer'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.51%, Threshold: 21.0, Value: 5.0
  - SIS Slots: 4 (Base: 4, Max: 4)
  - Leader Skill:
    - Main: Boosts 'Pure' by 9.0%
    - Extra: Boosts 'Pure' for 'second-year' by 6.0%

Deck ID: 4
<Card id=1000 name='Diving Mari' rarity='SR'>
  - Info: Character='Ohara Mari', Attribute='Smile', Level=60, Idolized=False
  - Stats (S/P/C): 4780/3130/3680
  - Skill: Level=1, Type='Perfect Lock'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.36%, Threshold: 26.0, Duration: 3.5s
  - SIS Slots: 2 (Base: 2, Max: 2)
  - Leader Skill:
    - Main: Boosts 'Smile' by 6.0%

Total cards in deck: 3
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_invalid_deck_id(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000, idolized=False, skill_level=4)
        with self.assertWarns(UserWarning) as cm:
            test_deck.remove_card(2)
        self.assertEqual(str(cm.warnings[0].message), "Deck ID 2 not found for removal.")
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=1000 name='Diving Mari' rarity='SR'>
  - Info: Character='Ohara Mari', Attribute='Smile', Level=60, Idolized=False
  - Stats (S/P/C): 4780/3130/3680
  - Skill: Level=4, Type='Perfect Lock'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.45%, Threshold: 26.0, Duration: 5.0s
  - SIS Slots: 2 (Base: 2, Max: 2)
  - Leader Skill:
    - Main: Boosts 'Smile' by 6.0%

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

        # Create a new deck, populate it, and then load over it
        test_deck = Deck(self.factory)
        test_deck.add_card(1500)
        test_deck.add_card(1600)
        test_deck.load_deck(self.DECK_SAVE_PATH)
        print(test_deck)
        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck ID: 1
<Card id=1000 name='Diving Mari' rarity='SR'>
  - Info: Character='Ohara Mari', Attribute='Smile', Level=60, Idolized=False
  - Stats (S/P/C): 4780/3130/3680
  - Skill: Level=1, Type='Perfect Lock'
    - Details: Activation: 'Rhythm Icons'
    - Effects: Chance: 0.36%, Threshold: 26.0, Duration: 3.5s
  - SIS Slots: 2 (Base: 2, Max: 2)
  - Leader Skill:
    - Main: Boosts 'Smile' by 6.0%

Total cards in deck: 1
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_delete_deck(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000)
        test_deck.add_card(1500)
        test_deck.delete_deck()
        print(test_deck)

        expected = """--- Current Deck Contents (Gallery Bonus: S/P/C 0/0/0) ---
Deck is currently empty.
--------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_update_gallery(self):
        test_deck = Deck(self.factory)
        test_deck.add_card(1000)

        old_smile = test_deck.get_card(1).stats.smile
        old_pure = test_deck.get_card(1).stats.pure
        old_cool = test_deck.get_card(1).stats.cool

        new_gallery = Gallery(5000, 3000, 2000)
        test_deck.gallery = new_gallery

        new_smile = test_deck.get_card(1).stats.smile
        new_pure = test_deck.get_card(1).stats.pure
        new_cool = test_deck.get_card(1).stats.cool

        self.assertEqual(old_smile + new_gallery.smile, new_smile)
        self.assertEqual(old_pure + new_gallery.pure, new_pure)
        self.assertEqual(old_cool + new_gallery.cool, new_cool)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
