import unittest
import os

from src.simulator.song.song_factory import SongFactory


class TestAccessory(unittest.TestCase):
    TEST_DIR = "data"
    JSON_PATH = os.path.join(TEST_DIR, "songs.json")

    def setUp(self):
        """Create a new factory instance for each test."""
        self.maxDiff = None

        self.factory = SongFactory(songs_json_path=self.JSON_PATH)

    def test_create_song_from_id(self):
        test_song = self.factory.create_song("Live_0007")

        expected_repr = """<Song id='Live_0007' title='Snow Halation'>
  - Difficulty: Easy
  - Group: μ's
  - Attribute: Cool
  - Length: 106.463s
  - Note Count: 81
  - Notes:
    - Note(start_time=2.07, end_time=2.07, position=7, is_star=False, is_swing=False)
    - Note(start_time=7.619, end_time=13.167, position=1, is_star=False, is_swing=False)
    - Note(start_time=14.556, end_time=14.556, position=8, is_star=False, is_swing=False)
    - Note(start_time=20.105, end_time=20.105, position=2, is_star=False, is_swing=False)
    - Note(start_time=24.267, end_time=24.267, position=6, is_star=False, is_swing=False)
    - Note(start_time=24.96, end_time=24.96, position=4, is_star=False, is_swing=False)
    - Note(start_time=25.654, end_time=25.654, position=1, is_star=False, is_swing=False)
    - Note(start_time=27.041, end_time=27.041, position=2, is_star=False, is_swing=False)
    - Note(start_time=28.428, end_time=28.428, position=3, is_star=False, is_swing=False)
    - Note(start_time=29.816, end_time=29.816, position=9, is_star=False, is_swing=False)
    - ...and 71 more entries."""

        self.assertEqual(str(test_song), expected_repr)

    def test_create_song_from_tuple(self):
        test_song = self.factory.create_song(("Snow Halation", "Master"))

        expected_repr = """<Song id='Live_s1548' title='Snow Halation'>
  - Difficulty: Master
  - Group: μ's
  - Attribute: Cool
  - Length: 109.932s
  - Note Count: 525
  - Notes:
    - Note(start_time=2.071, end_time=2.071, position=2, is_star=False, is_swing=False)
    - Note(start_time=2.071, end_time=2.071, position=7, is_star=False, is_swing=False)
    - Note(start_time=2.244, end_time=2.244, position=6, is_star=False, is_swing=False)
    - Note(start_time=2.418, end_time=2.418, position=7, is_star=False, is_swing=False)
    - Note(start_time=2.591, end_time=2.591, position=8, is_star=False, is_swing=False)
    - Note(start_time=2.764, end_time=2.764, position=9, is_star=False, is_swing=False)
    - Note(start_time=3.111, end_time=3.111, position=7, is_star=False, is_swing=False)
    - Note(start_time=3.285, end_time=3.285, position=6, is_star=False, is_swing=False)
    - Note(start_time=3.458, end_time=3.458, position=4, is_star=False, is_swing=False)
    - Note(start_time=3.632, end_time=3.632, position=3, is_star=False, is_swing=False)
    - ...and 515 more entries."""

        self.assertEqual(str(test_song).strip(), expected_repr)

    def test_access_note_attributes(self):
        test_song = self.factory.create_song(("Snow Halation", "Master"))

        self.assertEqual(test_song.notes[5].position, 9)
