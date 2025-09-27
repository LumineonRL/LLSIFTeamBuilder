use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct AccessoryStats {
    pub smile: u32,
    pub pure: u32,
    pub cool: u32,
}

impl AccessoryStats {
    pub fn new(smile: u32, pure: u32, cool: u32) -> Self {
        Self { smile, pure, cool }
    }
}
