use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Extra {
    #[serde(default, alias = "leader_extra_attribute")]
    extra_attribute: Option<String>,
    #[serde(default, alias = "leader_extra_target")]
    extra_target: Option<String>,
    #[serde(default, alias = "leader_extra_value")]
    extra_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeaderSkill {
    #[serde(alias = "leader_attribute")]
    pub attribute: Option<String>,
    #[serde(alias = "leader_secondary_attribute")]
    pub secondary_attribute: Option<String>,
    #[serde(default, alias = "leader_value")]
    pub value: Option<f64>,
    #[serde(flatten)]
    extra: Extra,
}

impl LeaderSkill {
    pub fn extra_attribute(&self) -> Option<&String> {
        self.extra.extra_attribute.as_ref()
    }

    pub fn extra_target(&self) -> Option<&String> {
        self.extra.extra_target.as_ref()
    }

    pub fn extra_value(&self) -> f64 {
        self.extra.extra_value.unwrap_or(0.0)
    }
}
