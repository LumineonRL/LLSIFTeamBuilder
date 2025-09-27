use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use crate::core::leader_skill::LeaderSkill;
use crate::core::skill::{Number, Skill};

use super::card_data::CardData;
use super::gallery::Gallery;
use super::stats::Stats;

pub struct Card {
    pub card_data: Arc<CardData>,
    gallery: Arc<RwLock<Gallery>>,

    pub idolized_status: String,
    base_stats: Stats,
    pub skill: Skill,
    pub leader_skill: LeaderSkill,

    pub level: u32,
    pub level_cap: u32,
    skill_level: u8,
    current_sis_slots: u8,
}

impl Card {
    pub fn new(
        card_data: Arc<CardData>,
        gallery: Arc<RwLock<Gallery>>,
        level_cap_map: &HashMap<String, serde_json::Value>,
        level_cap_bonus_map: &HashMap<String, serde_json::Value>,
        idolized: bool,
        level: Option<u32>,
    ) -> Result<Self, String> {
        let idolized_status = if idolized {
            "idolized".to_string()
        } else {
            "unidolized".to_string()
        };

        // Initialize base_stats from card_data
        let mut base_stats = card_data
            .stats
            .get(&idolized_status)
            .cloned()
            .ok_or_else(|| format!("Stats not found for status: {idolized_status}"))?;

        let skill = card_data.skill.clone();
        let leader_skill = card_data.leader_skill.clone();

        // Determine level and level_cap
        let level_cap = level_cap_map
            .get(&card_data.rarity)
            .and_then(|rarity_map| rarity_map.get(&idolized_status))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        let mut final_level = level_cap;
        if card_data.rarity == "UR"
            && idolized_status == "idolized"
            && let Some(l) = level
        {
            if (100..=500).contains(&l) {
                final_level = l;
            } else {
                println!(
                    "Custom level {l} for UR is out of range (100-500). Using default level cap."
                );
            }
        }

        // Apply level cap bonus
        let bonus_type = if card_data.is_promo && !card_data.is_preidolized_non_promo {
            "promo"
        } else {
            "non_promo"
        };

        if let Some(bonus_value) = level_cap_bonus_map
            .get(bonus_type)
            .and_then(|bonus_map| bonus_map.get(final_level.to_string()))
            .and_then(|v| v.as_u64())
        {
            let bonus_value = bonus_value as u32;
            base_stats.smile += bonus_value;
            base_stats.pure += bonus_value;
            base_stats.cool += bonus_value;
        }

        let initial_sis_slots = base_stats.sis_base;

        Ok(Self {
            card_data,
            gallery,
            idolized_status,
            base_stats,
            skill,
            leader_skill,
            level: final_level,
            level_cap,
            skill_level: 1,
            current_sis_slots: initial_sis_slots as u8,
        })
    }

    pub fn stats(&self) -> Stats {
        let gallery_bonus = self.gallery.read().unwrap();
        Stats {
            smile: self.base_stats.smile + gallery_bonus.smile,
            pure: self.base_stats.pure + gallery_bonus.pure,
            cool: self.base_stats.cool + gallery_bonus.cool,
            ..self.base_stats.clone()
        }
    }

    pub fn skill_level(&self) -> u8 {
        self.skill_level
    }

    pub fn set_skill_level(&mut self, value: u8) -> Result<(), String> {
        if !(1..=8).contains(&value) {
            return Err("Skill level must be between 1 and 8.".to_string());
        }
        self.skill_level = value;
        Ok(())
    }

    pub fn current_sis_slots(&self) -> u8 {
        self.current_sis_slots
    }

    pub fn set_current_sis_slots(&mut self, value: u8) -> Result<(), String> {
        let stats = self.stats();
        if !(stats.sis_base..=stats.sis_max).contains(&(value as u32)) {
            return Err(format!(
                "SIS slots must be between {} and {}.",
                stats.sis_base, stats.sis_max
            ));
        }
        self.current_sis_slots = value;
        Ok(())
    }

    fn get_skill_attribute_for_level<T: Clone>(&self, value_list: &[T], level: u8) -> Option<T> {
        if value_list.is_empty() {
            return None;
        }
        let index = (level - 1) as usize;
        let clamped_index = index.min(value_list.len() - 1);
        value_list.get(clamped_index).cloned()
    }

    pub fn skill_chance(&self) -> Option<f64> {
        self.get_skill_attribute_for_level(&self.skill.chances, self.skill_level)
    }

    pub fn skill_value(&self) -> Option<Number> {
        self.get_skill_attribute_for_level(&self.skill.values, self.skill_level)
            .flatten()
    }

    pub fn skill_threshold(&self) -> Option<i64> {
        self.get_skill_attribute_for_level(&self.skill.thresholds, self.skill_level)
            .flatten()
            .map(|n| match n {
                Number::Int(i) => i,
                Number::Float(f) => f as i64,
            })
    }

    pub fn skill_duration(&self) -> Option<Number> {
        self.get_skill_attribute_for_level(&self.skill.durations, self.skill_level)
            .flatten()
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        let ls = &self.leader_skill;

        let mut skill_details_parts = vec![];
        if let Some(act) = &self.skill.skill_type {
            skill_details_parts.push(format!("Activation: '{act}'"));
        }
        if let Some(t) = &self.skill.target {
            skill_details_parts.push(format!("Target: '{t}'"));
        }

        let mut skill_values_parts = vec![];
        if let Some(chance) = self.skill_chance() {
            skill_values_parts.push(format!("Chance: {chance}%"));
        }
        if let Some(threshold) = self.skill_threshold() {
            skill_values_parts.push(format!("Threshold: {threshold}"));
        }
        if let Some(value) = self.skill_value() {
            skill_values_parts.push(format!("Value: {value}"));
        }
        if let Some(duration) = self.skill_duration() {
            skill_values_parts.push(format!("Duration: {duration}s"));
        }

        let ls_main = format!(
            "Boosts '{}' {}by {:.1}%",
            ls.attribute.as_deref().unwrap_or("N/A"),
            ls.secondary_attribute
                .as_ref()
                .map(|s| format!("based on '{s}' "))
                .unwrap_or_default(),
            ls.value.unwrap_or(0.0) * 100.0
        );

        let ls_extra = if let (Some(attr), Some(target)) = (ls.extra_attribute(), ls.extra_target())
        {
            format!(
                "    - Extra: Boosts '{}' for '{}' by {:.1}%",
                attr,
                target,
                ls.extra_value() * 100.0
            )
        } else {
            "".to_string()
        };

        writeln!(
            f,
            "<Card id={} name='{}' rarity='{}'>",
            self.card_data.card_id, self.card_data.display_name, self.card_data.rarity
        )?;
        writeln!(
            f,
            "  - Info: Character='{}', Attribute='{}', Level={}, Idolized={}",
            self.card_data.character,
            self.card_data.attribute,
            self.level,
            self.idolized_status == "idolized"
        )?;
        writeln!(
            f,
            "  - Stats (S/P/C): {}/{}/{}",
            stats.smile, stats.pure, stats.cool
        )?;
        writeln!(
            f,
            "  - Skill: Level={}, Type='{}'",
            self.skill_level,
            self.skill.skill_type.as_deref().unwrap_or("N/A")
        )?;
        if !skill_details_parts.is_empty() {
            writeln!(f, "    - Details: {}", skill_details_parts.join(", "))?;
        }
        if !skill_values_parts.is_empty() {
            writeln!(f, "    - Effects: {}", skill_values_parts.join(", "))?;
        }
        writeln!(
            f,
            "  - SIS Slots: {} (Base: {}, Max: {})",
            self.current_sis_slots, stats.sis_base, stats.sis_max
        )?;
        writeln!(f, "  - Leader Skill:")?;
        writeln!(f, "    - Main: {ls_main}")?;
        if !ls_extra.is_empty() {
            writeln!(f, "{ls_extra}")?;
        }

        Ok(())
    }
}

impl PartialEq for Card {
    fn eq(&self, other: &Self) -> bool {
        self.card_data.card_id == other.card_data.card_id
            && self.idolized_status == other.idolized_status
            && self.level == other.level
            && self.skill_level == other.skill_level
    }
}
