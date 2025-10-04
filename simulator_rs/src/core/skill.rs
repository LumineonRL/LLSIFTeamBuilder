use serde::Deserialize;
use std::fmt;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Number {
    Int(i32),
    Float(f32),
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Int(i) => write!(f, "{i}"),
            Number::Float(fl) => write!(f, "{fl}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Skill {
    #[serde(rename = "type")]
    pub skill_type: Option<String>,
    pub activation: Option<String>,
    pub target: Option<String>,
    #[serde(default, alias = "level")]
    pub levels: Vec<Number>,
    #[serde(default, alias = "threshold")]
    pub thresholds: Vec<Option<Number>>,
    #[serde(default, alias = "chance")]
    pub chances: Vec<f32>,
    #[serde(default, alias = "value")]
    pub values: Vec<Option<Number>>,
    #[serde(default, alias = "duration")]
    pub durations: Vec<Option<Number>>,
}
