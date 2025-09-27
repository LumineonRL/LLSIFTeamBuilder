use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
pub struct Gallery {
    pub smile: u32,
    pub pure: u32,
    pub cool: u32,
}

impl Gallery {
    pub fn new(smile: u32, pure: u32, cool: u32) -> Self {
        Self { smile, pure, cool }
    }
}
