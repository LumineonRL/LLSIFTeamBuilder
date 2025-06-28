from typing import List

from src.simulator.song.note import Note
from src.simulator.song.song_data import SongData


class Song:
    """
    Represents a usable instance of a song, providing easy access to its
    attributes and a formatted representation.
    """

    def __init__(self, song_data: SongData):
        self._data = song_data

        self.song_id: str = self._data.song_id
        self.title: str = self._data.title
        self.difficulty: str = self._data.difficulty
        self.group: str = self._data.group
        self.attribute: str = self._data.attribute

        self.notes: List[Note] = [Note(**note_data) for note_data in self._data.notes]

    @property
    def length(self) -> float:
        """The total length of the song in seconds, based on the final note's endTime."""
        if not self.notes:
            return 0.0
        return max(note.end_time for note in self.notes)

    def __repr__(self) -> str:
        """Provides a detailed summary of the song."""
        header = f"<Song id='{self.song_id}' title='{self.title}'>"
        details = (
            f"  - Difficulty: {self.difficulty}\n"
            f"  - Group: {self.group}\n"
            f"  - Attribute: {self.attribute}\n"
            f"  - Length: {self.length:.3f}s\n"
            f"  - Note Count: {len(self.notes)}"
        )

        notes_summary = "\n  - Notes:"
        if not self.notes:
            notes_summary += " None"
        else:
            for _, note in enumerate(self.notes[:10]):
                notes_summary += f"\n    - {note}"
            if len(self.notes) > 10:
                notes_summary += f"\n    - ...and {len(self.notes) - 10} more entries."

        return f"{header}\n{details}{notes_summary}"
