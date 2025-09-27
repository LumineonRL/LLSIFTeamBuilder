use crate::core::skill::Number;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Default)]
pub struct EffectData {
    #[serde(rename = "type")]
    pub effect_type: Option<String>,
    #[serde(default)]
    pub durations: Vec<Option<Number>>,
    #[serde(default)]
    pub values: Vec<Option<Number>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Default)]
pub struct TriggerData {
    #[serde(default)]
    pub chances: Vec<f64>,
    #[serde(default)]
    pub values: Vec<Option<Number>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Default)]
pub struct SkillData {
    pub target: Option<String>,
    #[serde(default)]
    pub trigger: TriggerData,
    #[serde(default)]
    pub effect: EffectData,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct AccessoryDataRaw {
    pub name: String,
    pub character: String,
    #[serde(default)]
    pub card_id: Option<String>,
    #[serde(default)]
    pub stats: Vec<Vec<u32>>,
    #[serde(default)]
    pub skill: SkillData,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessoryData {
    pub accessory_id: u32,
    pub name: String,
    pub character: String,
    pub card_id: Option<String>,
    pub stats: Vec<Vec<u32>>,
    pub skill: SkillData,
}
