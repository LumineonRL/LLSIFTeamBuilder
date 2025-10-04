use crate::accessory::accessory_data::AccessoryData;
use crate::accessory::stats::AccessoryStats;
use crate::core::skill::{Number, Skill};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Accessory {
    pub data: Arc<AccessoryData>,
    pub skill_level: u8,
    pub stats: AccessoryStats,
    pub skill: Skill,
}

impl Accessory {
    pub fn new(data: Arc<AccessoryData>, skill_level: u8) -> Result<Self, String> {
        if !(1..=8).contains(&skill_level) {
            return Err(format!(
                "Accessory skill_level must be between 1 and 8, but got {skill_level}"
            ));
        }

        let skill_data = &data.skill;

        let skill = Skill {
            skill_type: skill_data.effect.effect_type.clone(),
            target: skill_data.target.clone(),
            chances: skill_data.trigger.chances.clone(),
            thresholds: skill_data.trigger.values.clone(),
            durations: skill_data.effect.durations.clone(),
            values: skill_data.effect.values.clone(),
            activation: None,
            levels: vec![],
        };

        let mut accessory = Self {
            data: data.clone(),
            skill_level: 1, // Placeholder, will be set correctly by set_skill_level
            stats: AccessoryStats::default(),
            skill,
        };

        accessory.set_skill_level(skill_level)?;
        Ok(accessory)
    }

    pub fn set_skill_level(&mut self, value: u8) -> Result<(), String> {
        if !(1..=8).contains(&value) {
            return Err(format!(
                "Accessory skill_level must be between 1 and 8, but got {value}"
            ));
        }
        self.skill_level = value;
        self.update_stats_from_skill_level();
        Ok(())
    }

    fn update_stats_from_skill_level(&mut self) {
        let index = (self.skill_level - 1) as usize;
        let clamped_index = index.min(self.data.stats.len().saturating_sub(1));

        let raw_stats = self
            .data
            .stats
            .get(clamped_index)
            .cloned()
            .unwrap_or_else(|| vec![0, 0, 0]);
        self.stats = AccessoryStats::new(
            raw_stats.first().copied().unwrap_or(0),
            raw_stats.get(1).copied().unwrap_or(0),
            raw_stats.get(2).copied().unwrap_or(0),
        );
    }

    fn get_skill_attribute_for_level<T: Clone>(&self, value_list: &[T], level: u8) -> Option<T> {
        if value_list.is_empty() {
            return None;
        }
        let index = (level - 1) as usize;
        let clamped_index = index.min(value_list.len() - 1);
        value_list.get(clamped_index).cloned()
    }

    pub fn skill_chance(&self) -> Option<f32> {
        self.get_skill_attribute_for_level(&self.skill.chances, self.skill_level)
    }

    pub fn skill_value(&self) -> Option<Number> {
        self.get_skill_attribute_for_level(&self.skill.values, self.skill_level)
            .flatten()
    }

    pub fn skill_threshold(&self) -> Option<u16> {
        self.get_skill_attribute_for_level(&self.skill.thresholds, self.skill_level)
            .flatten()
            .and_then(|n| match n {
                Number::Int(i) => i.try_into().ok(),
                _ => None,
            })
    }

    pub fn skill_duration(&self) -> Option<f32> {
        self.get_skill_attribute_for_level(&self.skill.durations, self.skill_level)
            .flatten()
            .map(|n| match n {
                Number::Float(f) => f,
                Number::Int(i) => i as f32,
            })
    }
}

impl PartialEq for Accessory {
    fn eq(&self, other: &Self) -> bool {
        self.data.accessory_id == other.data.accessory_id && self.skill_level == other.skill_level
    }
}

impl Display for Accessory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let header = format!(
            "<Accessory id={} name='{}' skill_level={}>",
            self.data.accessory_id, self.data.name, self.skill_level
        );
        let stats_line = format!(
            "  - Stats: Smile={}, Pure={}, Cool={}",
            self.stats.smile, self.stats.pure, self.stats.cool
        );

        let mut skill_lines = Vec::new();
        if let Some(skill_type) = &self.skill.skill_type {
            skill_lines.push(format!("  - Skill: Type='{skill_type}'"));

            let mut skill_details_parts = Vec::new();
            if let Some(target) = &self.skill.target {
                skill_details_parts.push(format!("Target: '{target}'"));
            }
            let skill_details = skill_details_parts.join(", ");
            if !skill_details.is_empty() {
                skill_lines.push(format!("    - Details: {skill_details}"));
            }

            let mut skill_values_parts = Vec::new();
            if let Some(chance) = self.skill_chance() {
                skill_values_parts.push(format!("Chance: {chance}%"));
            }
            if let Some(threshold) = self.skill_threshold() {
                skill_values_parts.push(format!("Threshold: {threshold}"));
            }
            if let Some(value) = self.skill_value() {
                let value_str = match value {
                    Number::Int(i) => i.to_string(),
                    Number::Float(fl) => fl.to_string(),
                };
                skill_values_parts.push(format!("Value: {value_str}"));
            }
            if let Some(duration) = self.skill_duration() {
                skill_values_parts.push(format!("Duration: {duration}s"));
            }
            let skill_values = skill_values_parts.join(", ");
            if !skill_values.is_empty() {
                skill_lines.push(format!("    - Effects: {skill_values}"));
            }
        }

        let mut all_lines = vec![header, stats_line];
        all_lines.extend(skill_lines);

        write!(f, "{}", all_lines.join("\n"))
    }
}
