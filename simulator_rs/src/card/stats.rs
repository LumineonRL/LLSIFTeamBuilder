use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Stats {
    pub smile: u32,
    pub pure: u32,
    pub cool: u32,
    #[serde(default = "default_sis_base")]
    pub sis_base: u32,
    #[serde(default = "default_sis_max")]
    pub sis_max: u32,
    pub image: Option<String>,
}

fn default_sis_base() -> u32 {
    1
}

fn default_sis_max() -> u32 {
    1
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            smile: 0,
            pure: 0,
            cool: 0,
            sis_base: 1,
            sis_max: 1,
            image: None,
        }
    }
}
