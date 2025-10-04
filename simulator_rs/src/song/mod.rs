use crate::song::note::Note;
use crate::song::song_data::SongData;
use std::fmt::{self, Display};

pub mod factory;
pub mod note;
pub mod song_data;

#[derive(Clone, Debug, PartialEq)]
pub struct Song {
    pub song_id: String,
    pub title: String,
    pub difficulty: String,
    pub group: String,
    pub attribute: String,
    pub notes: Vec<Note>,
}

impl Song {
    pub fn new(song_data: SongData) -> Self {
        Self {
            song_id: song_data.song_id,
            title: song_data.title,
            difficulty: song_data.difficulty,
            group: song_data.group,
            attribute: song_data.attribute,
            notes: song_data.notes,
        }
    }

    pub fn length(&self) -> f64 {
        self.notes
            .iter()
            .map(|note| note.end_time)
            .fold(0.0, f64::max)
    }
}

impl Display for Song {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = format!("<Song id='{}' title='{}'>", self.song_id, self.title);
        let details = format!(
            "  - Difficulty: {}\n  - Group: {}\n  - Attribute: {}\n  - Length: {:.3}s\n  - Note Count: {}",
            self.difficulty,
            self.group,
            self.attribute,
            self.length(),
            self.notes.len()
        );

        let mut notes_summary = "\n  - Notes:".to_string();
        if self.notes.is_empty() {
            notes_summary.push_str(" None");
        } else {
            for note in self.notes.iter().take(10) {
                notes_summary.push_str(&format!("\n    - {note}"));
            }
            if self.notes.len() > 10 {
                notes_summary.push_str(&format!(
                    "\n    - ...and {} more entries.",
                    self.notes.len() - 10
                ));
            }
        }

        write!(f, "{header}\n{details}{notes_summary}")
    }
}
