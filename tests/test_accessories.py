import unittest
import os

from accessoryfactory import AccessoryFactory

class TestAccessory(unittest.TestCase):

    TEST_DIR = "data"
    JSON_PATH = os.path.join(TEST_DIR, "accessories.json")

    def setUp(self):
        """Create a new factory instance for each test."""
        self.factory = AccessoryFactory(accessories_json_path=self.JSON_PATH)

    def test_create_accessory(self):
        """Verify an accessory at level 1 is created with the correct state."""
        test_accessory = self.factory.create_accessory(1)

        expected_repr = (
            "<Accessory id=1 card_id=165 name='UR Pink Ice Cream Diary' level=1>\n"
            "  - Stats: Smile=30, Pure=90, Cool=50\n"
            "  - Skill Type: Param up\n"
            "  - Trigger: 25% chance, Value: 0\n"
            "  - Effect Value: 20, Duration: 3.3s"
        )

        self.assertIsNotNone(test_accessory)
        self.assertEqual(str(test_accessory), expected_repr)

    def test_invalid_accessory(self):
        """Verify creating a non-existent accessory returns None and issues a warning."""
        with self.assertWarns(UserWarning) as cm:
            test_acc = self.factory.create_accessory(5000)

        self.assertIsNone(test_acc)
        self.assertIn("Error: Accessory with ID 5000 not found.", str(cm.warning))

    def test_accessory_level(self):
        """Verify an accessory at a specific level has the correct state."""
        test_accessory = self.factory.create_accessory(1, level=3)

        expected_repr = (
            "<Accessory id=1 card_id=165 name='UR Pink Ice Cream Diary' level=3>\n"
            "  - Stats: Smile=110, Pure=290, Cool=170\n"
            "  - Skill Type: Param up\n"
            "  - Trigger: 35% chance, Value: 0\n"
            "  - Effect Value: 26, Duration: 4.2s"
        )

        self.assertIsNotNone(test_accessory)
        self.assertEqual(str(test_accessory), expected_repr)

    def test_invalid_accessory_level(self):
        """Verify creating an accessory with an invalid level returns None and issues a warning."""
        with self.assertWarns(UserWarning) as cm:
            test_accessory = self.factory.create_accessory(1, level=10)

        self.assertIsNone(test_accessory)
        self.assertIn("Error creating accessory 1: Accessory level must be between 1 and 8.", str(cm.warning))

    def test_get_skill_for_arbitrary_level(self):
        """Test the public method for getting skill values at any valid level."""
        test_accessory = self.factory.create_accessory(315)
        self.assertIsNotNone(test_accessory)
        
        # Get skill value for level 3, even though the accessory's current level is 1
        chance_at_lvl_10 = test_accessory.get_skill_value_for_level(test_accessory.skill_trigger_chances, 10)
        self.assertEqual(chance_at_lvl_10, 40)

    def test_repr_output(self):
        """Verify the __repr__ method produces the correct, formatted output."""
        test_accessory = self.factory.create_accessory(1, level=2)
        expected_repr = (
            "<Accessory id=1 card_id=165 name='UR Pink Ice Cream Diary' level=2>\n"
            "  - Stats: Smile=70, Pure=190, Cool=110\n"
            "  - Skill Type: Param up\n"
            "  - Trigger: 30% chance, Value: 0\n"
            "  - Effect Value: 23, Duration: 3.8s"
        )
        self.assertEqual(str(test_accessory), expected_repr)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
