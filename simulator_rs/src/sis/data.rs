use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct SisData {
    pub id: u32,
    pub name: String,
    pub effect: String,
    pub slots: u32,
    pub attribute: String,
    pub group: Option<String>,
    pub equip_restriction: Option<String>,
    pub target: Option<String>,
    #[serde(default)]
    pub value: f64,
}
