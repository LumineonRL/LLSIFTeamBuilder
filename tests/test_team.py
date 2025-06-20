import unittest
import os
import warnings

from cardfactory import CardFactory
from deck import Deck
from accessoryfactory import AccessoryFactory
from accessorymanager import AccessoryManager
from sisfactory import SISFactory
from sismanager import SISManager
from guest import Guest
from team import Team

class TestTeam(unittest.TestCase):
    """Unit tests for the Team and TeamSlot classes."""

    @classmethod
    def setUpClass(cls):
        cls.tests_dir = os.path.dirname(__file__)
        cls.test_data_dir = os.path.join(cls.tests_dir, 'test_data')

    def setUp(self):
        """
        Initialize factories and managers for each test.
        This runs before every single test method.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, 'data')

        try:
            self.card_factory = CardFactory(
                cards_json_path=os.path.join(data_dir, 'cards.json'),
                level_caps_json_path=os.path.join(data_dir, 'level_caps.json'),
                level_cap_bonuses_path=os.path.join(data_dir, 'level_cap_bonuses.json')
            )
            self.accessory_factory = AccessoryFactory(os.path.join(data_dir, 'accessories.json'))
            self.sis_factory = SISFactory(os.path.join(data_dir, 'sis.json'))
        except RuntimeError as e:
            self.fail(f"FATAL: Could not load master data files for tests. {e}")

        self.deck = Deck(self.card_factory)
        self.accessory_manager = AccessoryManager(self.accessory_factory)
        self.sis_manager = SISManager(self.sis_factory)
        self.guest_manager = Guest(os.path.join(data_dir, 'unique_leader_skills.json'))

        self.deck.load_deck(os.path.join(self.test_data_dir, 'test_deck.json'))
        self.accessory_manager.load(os.path.join(self.test_data_dir, 'test_accs.json'))
        self.sis_manager.load(os.path.join(self.test_data_dir, 'test_sis.json'))

        self.team = Team(self.deck, self.accessory_manager, self.sis_manager, self.guest_manager)

        warnings.simplefilter("ignore", UserWarning)


    def test_empty_team(self):
        """Test the string representation of a newly created, empty team."""
        expected = """
--- Team Configuration ---
Guest: None
<Team is empty>
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_card_in_slot(self):
        """Test equipping a single card to a slot."""
        self.team.equip_card_in_slot(1, 1)
        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: None
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_card_in_oob_slot(self):
        """Test equipping a card to an out-of-bounds slot index."""
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_card_in_slot(10, 1)

        self.assertEqual(str(cm.warning), "Invalid slot number: 10. Must be between 1 and 9.")

        expected_team_state = """
--- Team Configuration ---
Guest: None
<Team is empty>
        """.strip()
        self.assertEqual(str(self.team).strip(), expected_team_state)

    def test_equip_card_oob_deck_id(self):
        """Test equipping a card with a non-existent deck ID."""
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_card_in_slot(1, 1000)

        self.assertEqual(str(cm.warning), "Card with Deck ID 1000 not found in the deck.")

        expected_team_state = """
--- Team Configuration ---
Guest: None
<Team is empty>
        """.strip()
        self.assertEqual(str(self.team).strip(), expected_team_state)

    def test_equip_card_overwrite_slot(self):
        """Test that equipping a new card to an occupied slot overwrites it."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_card_in_slot(1, 6)

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: 2774 Emma (Deck ID: 6)
  Accessory: None
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_accessory(self):
        """Test equipping an accessory to a card."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_accessory_in_slot(1, 3)

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: SR Crystal Ring (Manager ID: 3)
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_accessory_no_card(self):
        """Test that an accessory cannot be equipped to a slot without a card."""
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_accessory_in_slot(1, 1)

        self.assertEqual(str(cm.warning), "Cannot equip accessory: No card in slot 1.")

        expected_team_state = """
--- Team Configuration ---
Guest: None
<Team is empty>
        """.strip()
        self.assertEqual(str(self.team).strip(), expected_team_state)

    def test_swap_accessory(self):
        """Test that equipping a new accessory overwrites the old one."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_accessory_in_slot(1, 3)
        self.team.equip_accessory_in_slot(1, 6) # This should overwrite acc 3

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: R Rose Brooch (Manager ID: 6)
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_change_card_removes_acc(self):
        """Test that changing a card in a slot removes the previously equipped accessory."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_accessory_in_slot(1, 3)
        self.team.equip_card_in_slot(1, 50)

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Kunoichi Honoka (Deck ID: 50)
  Accessory: None
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_acc_oob_acc_id(self):
        """Test equipping an accessory with a non-existent manager ID."""
        self.team.equip_card_in_slot(1, 1)
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_accessory_in_slot(1, 1000)

        self.assertEqual(str(cm.warning), "Accessory with Manager ID 1000 not found.")

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: None
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_sis(self):
        """Test equipping a single SIS to a card."""
        self.team.equip_card_in_slot(1, 1) # Card has 4 SIS slots
        self.team.equip_sis_in_slot(1, 33)   # Pure Kiss, 1 slot

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: None
  SIS (1/4 slots used):
    - Pure Kiss (1 slots)
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_sis_no_card(self):
        """Test that a SIS cannot be equipped to a slot without a card."""
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_sis_in_slot(1, 33)

        self.assertEqual(str(cm.warning), "Cannot equip SIS: No card in slot 1.")

    def test_equip_sis_multiple_times(self):
        """Test that the same SIS instance cannot be equipped twice."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_sis_in_slot(1, 33)
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_sis_in_slot(1, 33) # Attempt to equip again

        self.assertEqual(str(cm.warning), "SIS with Manager ID 33 is already assigned.")

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: None
  SIS (1/4 slots used):
    - Pure Kiss (1 slots)
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_equip_sis_slot_overflow(self):
        """Test that a SIS cannot be equipped if there are not enough slots."""
        self.team.equip_card_in_slot(1, 1) # Card has 4 slots
        self.team.equip_sis_in_slot(1, 33)   # Pure Kiss, 1 slot. 3 slots remaining.
        with self.assertWarns(UserWarning) as cm:
            self.team.equip_sis_in_slot(1, 64) # Smile Fortune Grand, 5 slots. Fails.

        self.assertEqual(str(cm.warning), "Cannot equip SIS 'Smile Fortune Grand': Not enough slots. (5 required, 3 available)")

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Make-Up Magic Nozomi (Deck ID: 1)
  Accessory: None
  SIS (1/4 slots used):
    - Pure Kiss (1 slots)
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)

    def test_change_card_unequips_sis(self):
        """Test that changing a card unequips all SIS from the slot."""
        self.team.equip_card_in_slot(1, 1)
        self.team.equip_sis_in_slot(1, 33)
        self.team.equip_card_in_slot(1, 64)

        expected = """
--- Team Configuration ---
Guest: None

[ Slot 1 ]
  Card: Fruits (Swimsuit) Nozomi (Deck ID: 64)
  Accessory: None
  SIS: None
--------------------------
        """.strip()
        self.assertEqual(str(self.team).strip(), expected)


if __name__ == '__main__':
    unittest.main()
