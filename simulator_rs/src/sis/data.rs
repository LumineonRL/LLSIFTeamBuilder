use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct SisData {
    pub id: u16,
    pub name: String,
    pub effect: String,
    pub slots: u8,
    pub attribute: String,
    pub group: Option<String>,
    pub equip_restriction: Option<String>,
    pub target: Option<String>,
    #[serde(default)]
    pub value: f32,
}
