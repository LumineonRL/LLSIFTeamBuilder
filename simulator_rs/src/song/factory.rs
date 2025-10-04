use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use log;
use rand::seq::SliceRandom;
use serde_json::Value;
use thiserror::Error;

use crate::song::Song;
use crate::song::song_data::SongData;

#[derive(Debug, Error)]
pub enum SongFactoryError {
    #[error("Failed to load or parse JSON from {path}: {source}")]
    JsonError {
        path: String,
        source: serde_json::Error,
    },
    #[error("IO error while reading from {path}: {source}")]
    IoError {
        path: String,
        source: std::io::Error,
    },
    #[error("Songs data file must be a dictionary of objects.")]
    InvalidRootType,
}

pub type SongIdentifier = (String, String);

pub struct SongFactory {
    song_data_by_id: HashMap<String, SongData>,
    song_data_by_title_diff: HashMap<SongIdentifier, SongData>,
}

impl SongFactory {
    pub fn new(json_path: &str) -> Result<Self, SongFactoryError> {
        let raw_data = Self::load_json(json_path)?;
        let (song_data_by_id, song_data_by_title_diff) = Self::index_song_data(raw_data);

        Ok(Self {
            song_data_by_id,
            song_data_by_title_diff,
        })
    }

    fn load_json(json_path: &str) -> Result<HashMap<String, Value>, SongFactoryError> {
        let path = Path::new(json_path);
        let file = File::open(path).map_err(|e| SongFactoryError::IoError {
            path: json_path.to_string(),
            source: e,
        })?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).map_err(|e| SongFactoryError::JsonError {
            path: json_path.to_string(),
            source: e,
        })
    }

    fn index_song_data(
        raw_data: HashMap<String, Value>,
    ) -> (HashMap<String, SongData>, HashMap<SongIdentifier, SongData>) {
        let mut by_id = HashMap::new();
        let mut by_title_diff = HashMap::new();

        for (song_id_json, record) in raw_data {
            let song_id = song_id_json.replace(".json", "");
            let mut data_instance: SongData = match serde_json::from_value(record) {
                Ok(data) => data,
                Err(e) => {
                    log::warn!(
                        "Warning: Skipping invalid song record with key '{song_id_json}': {e}"
                    );
                    continue;
                }
            };
            data_instance.song_id = song_id.clone();

            let title_diff_key = (
                data_instance.title.clone(),
                data_instance.difficulty.clone(),
            );

            by_id.insert(song_id, data_instance.clone());
            by_title_diff.insert(title_diff_key, data_instance);
        }

        (by_id, by_title_diff)
    }

    pub fn create_song_by_id(&self, song_id: &str) -> Option<Song> {
        self.song_data_by_id
            .get(song_id)
            .map(|data| Song::new(data.clone()))
    }

    pub fn create_song_by_title_diff(&self, title: &str, difficulty: &str) -> Option<Song> {
        let key = &(title.to_string(), difficulty.to_string());
        self.song_data_by_title_diff
            .get(key)
            .map(|data| Song::new(data.clone()))
    }

    pub fn get_random_identifier(&self) -> Option<String> {
        let all_ids: Vec<&String> = self.song_data_by_id.keys().collect();
        all_ids
            .choose(&mut rand::thread_rng())
            .map(|id| (*id).clone())
    }
}
