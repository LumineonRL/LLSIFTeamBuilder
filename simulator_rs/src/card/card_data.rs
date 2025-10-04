use serde::Deserialize;
use std::collections::HashMap;

use crate::core::leader_skill::LeaderSkill;
use crate::core::skill::Skill;

use super::stats::Stats;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CardData {
    pub card_id: u16,
    pub display_name: String,
    pub rarity: String,
    pub attribute: String,
    pub character: String,
    pub is_promo: bool,
    pub is_preidolized_non_promo: bool,
    pub stats: HashMap<String, Stats>,
    pub skill: Skill,
    pub leader_skill: LeaderSkill,
}
