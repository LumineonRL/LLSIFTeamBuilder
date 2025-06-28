import unittest
import os

from src.simulator.sis.sis_factory import SISFactory


class TestAccessory(unittest.TestCase):

    TEST_DIR = "data"
    JSON_PATH = os.path.join(TEST_DIR, "sis.json")

    def setUp(self):
        """Create a new factory instance for each test."""
        self.factory = SISFactory(sis_json_path=self.JSON_PATH)

    def test_create_sis(self):
        """Verify a SIS is created with the correct parameters."""
        test_sis = self.factory.create_sis(1)

        expected_repr = """<SIS id=1 name='Smile Kiss'>
  - Effect: self flat boost (200.0)
  - Slots: 1, Attribute: Smile"""

        self.assertIsNotNone(test_sis)
        self.assertEqual(str(test_sis), expected_repr)

    def test_invalid_sis(self):
        """Verify creating a non-existent SIS returns None and issues a warning."""
        with self.assertWarns(UserWarning) as cm:
            test_sis = self.factory.create_sis(5000)

        self.assertIsNone(test_sis)
        self.assertIn("Error: SIS with ID 5000 not found.", str(cm.warning))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
