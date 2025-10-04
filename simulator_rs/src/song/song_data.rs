use crate::song::note::Note;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct SongData {
    #[serde(skip_deserializing)]
    pub song_id: String,
    pub title: String,
    pub difficulty: String,
    pub group: String,
    pub attribute: String,
    #[serde(default)]
    pub notes: Vec<Note>,
}
