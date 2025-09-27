use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, RwLock};

use serde_json;

use super::card_data::CardData;
use super::core::Card;
use super::gallery::Gallery;

pub struct CardFactory {
    card_data_map: Arc<HashMap<u32, Arc<CardData>>>,
    level_cap_map: Arc<HashMap<String, serde_json::Value>>,
    level_cap_bonus_map: Arc<HashMap<String, serde_json::Value>>,
}

impl CardFactory {
    pub fn new(
        cards_json_path: &str,
        level_caps_json_path: &str,
        level_cap_bonuses_path: &str,
    ) -> Result<Self, String> {
        let card_data_map = Arc::new(Self::load_and_index_card_data(cards_json_path)?);
        let level_cap_map = Arc::new(Self::load_json(level_caps_json_path)?);
        let level_cap_bonus_map = Arc::new(Self::load_json(level_cap_bonuses_path)?);

        Ok(Self {
            card_data_map,
            level_cap_map,
            level_cap_bonus_map,
        })
    }

    fn load_json<T: for<'de> serde::Deserialize<'de>>(path: &str) -> Result<T, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse JSON from {path}: {e}"))
    }

    fn load_and_index_card_data(path: &str) -> Result<HashMap<u32, Arc<CardData>>, String> {
        let raw_data: Vec<CardData> = Self::load_json(path)?;
        let mut indexed_map = HashMap::new();
        for record in raw_data {
            indexed_map.insert(record.card_id, Arc::new(record));
        }
        Ok(indexed_map)
    }

    pub fn create_card(
        &self,
        card_id: u32,
        gallery: Arc<RwLock<Gallery>>,
        idolized: bool,
        skill_level: u8,
        level: Option<u32>,
        sis_slots: Option<u8>,
    ) -> Option<Card> {
        let card_data = self.card_data_map.get(&card_id)?;

        let final_idolized = if card_data.is_promo && !idolized {
            println!(
                "Card ID {card_id} is a promo and cannot be unidolized. Forcing to idolized state."
            );
            true
        } else {
            idolized
        };

        let mut card = Card::new(
            Arc::clone(card_data),
            gallery,
            &self.level_cap_map,
            &self.level_cap_bonus_map,
            final_idolized,
            level,
        )
        .map_err(|e| {
            println!("Failed to create card (ID: {card_id}): {e}");
            e
        })
        .ok()?;

        if let Err(e) = card.set_skill_level(skill_level) {
            println!("Invalid skill level for card ID {card_id}: {e}. Defaulting to 1.");
            // The card is already at skill level 1 by default.
        }

        if let Some(slots) = sis_slots
            && let Err(e) = card.set_current_sis_slots(slots)
        {
            println!("Invalid SIS slots for card ID {card_id}: {e}. Using default.");
        }

        Some(card)
    }
}
