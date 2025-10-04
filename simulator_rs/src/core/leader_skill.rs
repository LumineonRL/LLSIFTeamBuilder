use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Extra {
    #[serde(default, alias = "leader_extra_attribute")]
    extra_attribute: Option<String>,
    #[serde(default, alias = "leader_extra_target")]
    extra_target: Option<String>,
    #[serde(default, alias = "leader_extra_value")]
    extra_value: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LeaderSkill {
    #[serde(alias = "leader_attribute")]
    pub attribute: Option<String>,
    #[serde(alias = "leader_secondary_attribute")]
    pub secondary_attribute: Option<String>,
    #[serde(default, alias = "leader_value")]
    pub value: Option<f32>,
    #[serde(flatten)]
    extra: Extra,
}

impl LeaderSkill {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attribute: Option<String>,
        secondary_attribute: Option<String>,
        value: Option<f32>,
        extra_attribute: Option<String>,
        extra_target: Option<String>,
        extra_value: Option<f32>,
    ) -> Self {
        Self {
            attribute,
            secondary_attribute,
            value,
            extra: Extra {
                extra_attribute,
                extra_target,
                extra_value,
            },
        }
    }

    pub fn extra_attribute(&self) -> Option<&String> {
        self.extra.extra_attribute.as_ref()
    }

    pub fn extra_target(&self) -> Option<&String> {
        self.extra.extra_target.as_ref()
    }

    pub fn extra_value(&self) -> f32 {
        self.extra.extra_value.unwrap_or(0.0)
    }
}
