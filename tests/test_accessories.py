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

        expected_repr = ("""<Accessory id=1 name='UR Pink Ice Cream Diary' skill_level=1>
  - Stats: Smile=30, Pure=90, Cool=50
  - Skill: Type='Appeal Boost'
    - Details: Target: 'Kosaka Honoka'
    - Effects: Chance: 25%, Threshold: 0, Value: 20, Duration: 3.3s""")

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
        test_accessory = self.factory.create_accessory(1, skill_level=3)

        expected_repr = ("""<Accessory id=1 name='UR Pink Ice Cream Diary' skill_level=3>
  - Stats: Smile=110, Pure=290, Cool=170
  - Skill: Type='Appeal Boost'
    - Details: Target: 'Kosaka Honoka'
    - Effects: Chance: 35%, Threshold: 0, Value: 26, Duration: 4.2s"""
        )

        self.assertIsNotNone(test_accessory)
        self.assertEqual(str(test_accessory), expected_repr)

    def test_invalid_accessory_level(self):
        """Verify creating an accessory with an invalid level returns None and issues a warning."""
        with self.assertWarns(UserWarning) as cm:
            test_accessory = self.factory.create_accessory(1, skill_level=10)

        self.assertIsNone(test_accessory)
        self.assertIn("Error creating accessory 1: Accessory skill_level must be between 1 and 8.", str(cm.warning))

    def test_get_skill_for_arbitrary_level(self):
        """Test the public method for getting skill values at any valid level."""
        test_accessory = self.factory.create_accessory(315)
        self.assertIsNotNone(test_accessory)

        chance_at_lvl_10 = test_accessory.get_skill_attribute_for_level(test_accessory.skill.chances, 10)
        self.assertEqual(chance_at_lvl_10, 40)

    def test_repr_output(self):
        """Verify the __repr__ method produces the correct, formatted output."""
        test_accessory = self.factory.create_accessory(1, skill_level=2)
        expected_repr = ("""<Accessory id=1 name='UR Pink Ice Cream Diary' skill_level=2>
  - Stats: Smile=70, Pure=190, Cool=110
  - Skill: Type='Appeal Boost'
    - Details: Target: 'Kosaka Honoka'
    - Effects: Chance: 30%, Threshold: 0, Value: 23, Duration: 3.8s""")
        self.assertEqual(str(test_accessory), expected_repr)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
