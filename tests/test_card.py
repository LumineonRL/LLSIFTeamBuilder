import unittest

from cardfactory import CardFactory

class TestCardFactory(unittest.TestCase):

    def setUp(self):
        """Set up the factory for each test."""
        # This block initializes the factory using the dummy files.
        # To use your local files, you can uncomment the alternative block.
        self.factory = CardFactory(
            cards_json_path="./data/cards.json",
            level_caps_json_path="./data/level_caps.json",
            level_cap_bonuses_path="./data/level_cap_bonuses.json"
        )

    def test_initialize_card(self):
        test_card = self.factory.create_card(101)
        self.assertEqual(f"Created card: {test_card.display_name} ({test_card.rarity})", "Created card: Initial Nozomi (UR)")

    def test_check_stats(self):
        test_card = self.factory.create_card(101)
        self.assertEqual(f"Smile: {test_card.stats.smile}", "Smile: 3890")

    def test_initialize_idolized(self):
        test_card = self.factory.create_card(101, idolized=True)
        self.assertEqual(f"Idolization Status: {test_card.idolized_status}", "Idolization Status: idolized")
  
    def test_card_out_of_range(self):
        test_card = self.factory.create_card(11101)
        self.assertIsNone(test_card)

    def test_modify_ur_level(self):
        test_card = self.factory.create_card(101, idolized=True, level=200)
        self.assertEqual(f"Smile: {test_card.stats.smile}", "Smile: 5990")

    def test_unidolized_promo(self):
        with self.assertWarns(UserWarning) as cm:
            test_card = self.factory.create_card(96, idolized=False)
            self.assertEqual(str(cm.warnings[0].message), "Card ID 96 is a promo and cannot be unidolized. Forcing to idolized state.")
        self.assertIsNotNone(test_card)
        self.assertEqual(f"Created card: {test_card.display_name} ({test_card.rarity})", "Created card: Season 1 BD Bonus Maki (UR)")

    def test_modify_promo_level(self):
        test_card = self.factory.create_card(96, idolized=True, level=101)
        self.assertEqual(f"Smile: {test_card.stats.smile}", "Smile: 3925")

    def test_modify_skill_level(self):
        test_card = self.factory.create_card(101, skill_level=5)
        self.assertEqual(f"Skill Chance: {test_card.skill_chance}", "Skill Chance: 0.52")

    def test_skill_level_out_of_range(self):
        with self.assertWarns(UserWarning) as cm:
            test_card = self.factory.create_card(101, skill_level=20)
            self.assertEqual(str(cm.warnings[0].message), "Invalid skill level '20'. Must be between 1 and 8. Defaulting to 1.")
        self.assertEqual(f"Skill Chance: {test_card.skill_chance}", "Skill Chance: 0.36")

    def test_modify_sis_slots(self):
        test_card = self.factory.create_card(101, idolized=True, sis_slots=6)
        self.assertEqual(f"SIS Slots: {test_card.current_sis_slots}", "SIS Slots: 6")

    def test_sis_slots_out_of_range(self):
        with self.assertWarns(UserWarning) as cm:
            test_card = self.factory.create_card(101, idolized=True, sis_slots=3)
            self.assertEqual(str(cm.warnings[0].message), "Invalid SIS slots value '3'. Must be between 4 and 8. Defaulting to 4.")
        self.assertEqual(f"SIS Slots: {test_card.current_sis_slots}", "SIS Slots: 4")

    def test_non_integer_level(self):
        with self.assertWarns(UserWarning) as cm:
            test_card = self.factory.create_card(101, idolized=True, level=101.5)
            self.assertEqual(str(cm.warnings[0].message), "Invalid type for level: got float, expected int. Ignoring custom level.")
        self.assertEqual(f"Level: {test_card.level}", "Level: 100")

    def test_get_skill_for_arbitrary_level(self):
        """Test the public method for getting skill values at any valid level."""
        test_card = self.factory.create_card(1021, idolized=True, skill_level=5)
        self.assertIsNotNone(test_card)
        
        chance_at_lvl_11 = test_card.get_skill_attribute_for_level(test_card.skill.chances, 10)
        self.assertEqual(chance_at_lvl_11, 0.41)

    def test_card_repr(self):
        """Test the public method for getting skill values at any valid level."""
        test_card = self.factory.create_card(101)
        self.assertIsNotNone(test_card)
        
        expected = """
<Card id=101 name='Initial Nozomi' rarity='UR'>
  - Info: Character='Tojo Nozomi', Attribute='Pure', Level=80, Idolized=False
  - Stats (S/P/C): 3890/5500/4000
  - Skill: Level=1, Type='Scorer'
    - Details: Activation: 'Combo'
    - Effects: Chance: 0.36%, Threshold: 17.0, Value: 450.0
  - SIS Slots: 4 (Base: 4, Max: 4)
  - Leader Skill:
    - Main: Boosts 'Pure' by 9.0%
    - Extra: Boosts 'Pure' for 'μ's' by 3.0%
        """.strip()
        self.assertEqual(str(test_card).strip(), expected)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
