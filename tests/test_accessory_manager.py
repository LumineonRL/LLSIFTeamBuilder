import unittest
import os
import sys
from io import StringIO

from src.simulator.accessory.accessory_factory import AccessoryFactory
from src.simulator.accessory.accessory_manager import AccessoryManager


class TestAccessoryManager(unittest.TestCase):

    ACCESSORY_SAVE_PATH = "./data/test_accessories.json"

    def setUp(self):
        """Set up the factory, redirect stdout, and ensure clean file state for each test."""
        self.factory = AccessoryFactory("./data/accessories.json")
        self.held_stdout = sys.stdout
        sys.stdout = self.captured_output = StringIO()

        if os.path.exists(self.ACCESSORY_SAVE_PATH):
            os.remove(self.ACCESSORY_SAVE_PATH)

    def tearDown(self):
        """Restore stdout and clean up test files."""
        sys.stdout = self.held_stdout

        if os.path.exists(self.ACCESSORY_SAVE_PATH):
            os.remove(self.ACCESSORY_SAVE_PATH)

    def test_add_accessory_display_deck(self):
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(101, skill_level=4)
        test_accessories.add_accessory(28)
        test_accessories.add_accessory(96)
        print(test_accessories)
        expected = """<AccessoryManager (3 accessories)>
  - ID 1: <Accessory id=101 name='UR Police Badge Case' skill_level=4>
  - Stats: Smile=380, Pure=320, Cool=570
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 22%, Threshold: 0, Value: 30, Duration: 5.5s
  - ID 2: <Accessory id=28 name='SSR Amethyst Pendant' skill_level=1>
  - Stats: Smile=50, Pure=30, Cool=20
  - Skill: Type='Healer'
    - Effects: Chance: 5%, Threshold: 0, Value: 4, Duration: 0s
  - ID 3: <Accessory id=96 name='UR Peach Badge' skill_level=1>
  - Stats: Smile=100, Pure=70, Cool=50
  - Skill: Type='Skill Rate Up'
    - Effects: Chance: 5%, Threshold: 0, Value: 4, Duration: 2.5s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_add_non_existent_accessory(self):
        test_accessories = AccessoryManager(self.factory)
        with self.assertWarns(UserWarning) as cm:
            test_accessories.add_accessory(4001, skill_level=4)
        self.assertEqual(
            str(cm.warnings[0].message), "Error: Accessory with ID 4001 not found."
        )
        test_accessories.add_accessory(28)
        print(test_accessories)
        expected = """<AccessoryManager (1 accessories)>
  - ID 1: <Accessory id=28 name='SSR Amethyst Pendant' skill_level=1>
  - Stats: Smile=50, Pure=30, Cool=20
  - Skill: Type='Healer'
    - Effects: Chance: 5%, Threshold: 0, Value: 4, Duration: 0s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_modify_accessory(self):
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(100, skill_level=4)
        test_accessories.modify_accessory(manager_internal_id=1, skill_level=6)
        print(test_accessories)
        expected = """<AccessoryManager (1 accessories)>
  - ID 1: <Accessory id=100 name='UR Block Checked Mug' skill_level=6>
  - Stats: Smile=910, Pure=600, Cool=500
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 27%, Threshold: 0, Value: 31, Duration: 6s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_accessory(self):
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(100, skill_level=4)
        test_accessories.add_accessory(101, skill_level=4)
        test_accessories.add_accessory(102, skill_level=4)
        test_accessories.remove_accessory(2)
        test_accessories.add_accessory(100)
        print(test_accessories)
        expected = """<AccessoryManager (3 accessories)>
  - ID 1: <Accessory id=100 name='UR Block Checked Mug' skill_level=4>
  - Stats: Smile=570, Pure=380, Cool=310
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 22%, Threshold: 0, Value: 28, Duration: 5.5s
  - ID 3: <Accessory id=102 name='UR Dalmatian Bag' skill_level=4>
  - Stats: Smile=570, Pure=380, Cool=320
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 22%, Threshold: 0, Value: 32, Duration: 5.5s
  - ID 4: <Accessory id=100 name='UR Block Checked Mug' skill_level=1>
  - Stats: Smile=220, Pure=150, Cool=120
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 10%, Threshold: 0, Value: 20, Duration: 2.5s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_invalid_accessory_manager_id(self):
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(100, skill_level=4)
        with self.assertWarns(UserWarning) as cm:
            test_accessories.remove_accessory(2)
        self.assertEqual(
            str(cm.warnings[0].message),
            "Accessory Manager ID 2 does not exist and cannot be removed.",
        )
        print(test_accessories)
        expected = """<AccessoryManager (1 accessories)>
  - ID 1: <Accessory id=100 name='UR Block Checked Mug' skill_level=4>
  - Stats: Smile=570, Pure=380, Cool=310
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 22%, Threshold: 0, Value: 28, Duration: 5.5s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_save_accessories(self):
        test_accessories = AccessoryManager(self.factory)
        self.assertFalse(os.path.exists(self.ACCESSORY_SAVE_PATH))
        test_accessories.add_accessory(100)
        test_accessories.save(self.ACCESSORY_SAVE_PATH)
        self.assertTrue(os.path.exists(self.ACCESSORY_SAVE_PATH))

    def test_load_accessories(self):
        # First, create and save accessories to load from
        accessories_to_save = AccessoryManager(self.factory)
        accessories_to_save.add_accessory(100)
        accessories_to_save.save(self.ACCESSORY_SAVE_PATH)

        # Now, create a new accessory manager, populate it, and then load over it
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(150)
        test_accessories.add_accessory(160)
        test_accessories.load(self.ACCESSORY_SAVE_PATH)
        print(test_accessories)
        expected = """<AccessoryManager (1 accessories)>
  - ID 1: <Accessory id=100 name='UR Block Checked Mug' skill_level=1>
  - Stats: Smile=220, Pure=150, Cool=120
  - Skill: Type='Combo Bonus Up'
    - Effects: Chance: 10%, Threshold: 0, Value: 20, Duration: 2.5s"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_delete_accessories(self):
        test_accessories = AccessoryManager(self.factory)
        test_accessories.add_accessory(100)
        test_accessories.add_accessory(150)
        test_accessories.delete()
        print(test_accessories)
        self.assertEqual(
            self.captured_output.getvalue().strip(), "<AccessoryManager (empty)>"
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
