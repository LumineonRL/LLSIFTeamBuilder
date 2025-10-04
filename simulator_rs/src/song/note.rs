use serde::Deserialize;
use std::fmt::{self, Display};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Note {
    #[serde(rename = "startTime")]
    pub start_time: f64,
    #[serde(rename = "endTime")]
    pub end_time: f64,
    pub position: i32,
    #[serde(rename = "isStar")]
    pub is_star: bool,
    #[serde(rename = "isSwing")]
    pub is_swing: bool,
}

impl Display for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Note(start_time={:?}, end_time={:?}, position={}, is_star={}, is_swing={})",
            self.start_time,
            self.end_time,
            self.position,
            if self.is_star { "True" } else { "False" },
            if self.is_swing { "True" } else { "False" }
        )
    }
}
