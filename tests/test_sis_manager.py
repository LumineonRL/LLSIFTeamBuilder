import unittest
import os
import sys
from io import StringIO

from src.simulator.sis.sis_factory import SISFactory
from src.simulator.sis.sis_manager import SISManager


class TestSISManager(unittest.TestCase):

    SIS_SAVE_PATH = "./data/test_sis.json"

    def setUp(self):
        """Set up the factory, redirect stdout, and ensure clean file state for each test."""
        self.factory = SISFactory("./data/sis.json")
        self.held_stdout = sys.stdout
        sys.stdout = self.captured_output = StringIO()

        if os.path.exists(self.SIS_SAVE_PATH):
            os.remove(self.SIS_SAVE_PATH)

    def tearDown(self):
        """Restore stdout and clean up test files."""
        sys.stdout = self.held_stdout

        if os.path.exists(self.SIS_SAVE_PATH):
            os.remove(self.SIS_SAVE_PATH)

    def test_add_sis(self):
        test_sis = SISManager(self.factory)
        test_sis.add_sis(101)
        test_sis.add_sis(28)
        test_sis.add_sis(96)
        print(test_sis)
        expected = """<SISManager (3 skills)>
  - ID 1: Smile Dolphin (SID: 101)
  - ID 2: Smile Veil (SID: 28)
  - ID 3: Cool Galaxy (SID: 96)"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_add_non_existent_sis(self):
        test_sis = SISManager(self.factory)
        with self.assertWarns(UserWarning) as cm:
            test_sis.add_sis(4001)
        self.assertEqual(
            str(cm.warnings[0].message), "Error: SIS with ID 4001 not found."
        )
        test_sis.add_sis(28)
        print(test_sis)
        expected = """<SISManager (1 skills)>
  - ID 1: Smile Veil (SID: 28)"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_sis(self):
        test_sis = SISManager(self.factory)
        test_sis.add_sis(100)
        test_sis.add_sis(101)
        test_sis.add_sis(102)
        test_sis.remove_sis(2)
        test_sis.add_sis(100)
        print(test_sis)
        expected = """<SISManager (3 skills)>
  - ID 1: Pure Blossom (SID: 100)
  - ID 3: Cool Dolphin (SID: 102)
  - ID 4: Pure Blossom (SID: 100)"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_remove_invalid_sis_manager_id(self):
        test_sis = SISManager(self.factory)
        test_sis.add_sis(100)
        with self.assertWarns(UserWarning) as cm:
            test_sis.remove_sis(2)
        self.assertEqual(
            str(cm.warnings[0].message),
            "SIS Manager ID 2 does not exist and cannot be removed.",
        )
        print(test_sis)
        expected = """<SISManager (1 skills)>
  - ID 1: Pure Blossom (SID: 100)"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_save_sis(self):
        test_sis = SISManager(self.factory)
        self.assertFalse(os.path.exists(self.SIS_SAVE_PATH))
        test_sis.add_sis(100)
        test_sis.save(self.SIS_SAVE_PATH)
        self.assertTrue(os.path.exists(self.SIS_SAVE_PATH))

    def test_load_sis(self):
        sis_to_save = SISManager(self.factory)
        sis_to_save.add_sis(100)
        sis_to_save.save(self.SIS_SAVE_PATH)

        test_sis = SISManager(self.factory)
        test_sis.add_sis(150)
        test_sis.add_sis(160)
        test_sis.load(self.SIS_SAVE_PATH)
        print(test_sis)
        expected = """<SISManager (1 skills)>
  - ID 1: Pure Blossom (SID: 100)"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_delete_sis(self):
        test_sis = SISManager(self.factory)
        test_sis.add_sis(100)
        test_sis.add_sis(150)
        test_sis.delete()
        print(test_sis)
        self.assertEqual(
            self.captured_output.getvalue().strip(), "<SISManager (empty)>"
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
