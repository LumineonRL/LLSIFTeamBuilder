import unittest
import sys
from io import StringIO

from src.simulator.team.guest import Guest


class TestGuest(unittest.TestCase):

    GUEST_FILE_PATH = "./data/unique_leader_skills.json"

    def setUp(self):
        """Set up the guest manager and redirect stdout for each test."""
        self.guest_manager = Guest(self.GUEST_FILE_PATH)
        self.held_stdout = sys.stdout
        sys.stdout = self.captured_output = StringIO()

    def tearDown(self):
        """Restore stdout."""
        sys.stdout = self.held_stdout

    def test_set_guest(self):
        self.guest_manager.set_guest(25)
        print(self.guest_manager)
        expected = """--- Current Guest Details ---
Leader Skill Id: 25
Leader Attribute: Smile
Leader Secondary Attribute: N/A
Leader Value: 0.09
Leader Extra Attribute: Smile
Leader Extra Target: Liella!
Leader Extra Value: 0.03
---------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_unset_guest(self):
        print(self.guest_manager)
        self.assertEqual(
            self.captured_output.getvalue().strip(), "<Guest active_id=None>"
        )

    def test_invalid_guest(self):
        with self.assertWarns(UserWarning) as cm:
            self.guest_manager.set_guest(25.9)
        self.assertEqual(str(cm.warnings[0].message), "Guest with ID 25.9 not found.")

        print(self.guest_manager)
        self.assertEqual(
            self.captured_output.getvalue().strip(), "<Guest active_id=None>"
        )

    def test_change_guest(self):
        self.guest_manager.set_guest(1)
        self.guest_manager.set_guest(2)
        self.guest_manager.set_guest(3)
        print(self.guest_manager)
        expected = """--- Current Guest Details ---
Leader Skill Id: 3
Leader Attribute: Smile
Leader Secondary Attribute: N/A
Leader Value: 0.04
Leader Extra Attribute: N/A
Leader Extra Target: N/A
Leader Extra Value: N/A
---------------------------"""
        self.assertEqual(self.captured_output.getvalue().strip(), expected)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
