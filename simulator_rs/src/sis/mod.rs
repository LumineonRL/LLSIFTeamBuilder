pub mod manager;
pub mod sis_factory;

pub use manager::PlayerSIS;
pub use manager::SISManager;
pub use sis_factory::SISFactory;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SIS {
    pub id: u32,
    pub name: String,
    pub effect: String,
    pub slots: u32,
    pub attribute: String,
    pub group: String,
    #[serde(rename = "equip_restriction")]
    pub equip_restriction: String,
    pub target: String,
    pub value: f32,
}
