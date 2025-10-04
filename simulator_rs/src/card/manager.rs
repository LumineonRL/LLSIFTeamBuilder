use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use super::card_factory::CardFactory;
use super::core::Card;
use super::gallery::Gallery;

#[derive(Debug, PartialEq, Clone)]
pub struct PlayerCard {
    pub manager_internal_id: u32,
    pub card: Card,
}

pub struct CardManager {
    card_factory: Arc<CardFactory>,
    cards: HashMap<u32, PlayerCard>,
    next_manager_internal_id: u32,
    gallery: Arc<RwLock<Gallery>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CardManagerState {
    #[serde(rename = "next_deck_id")]
    next_manager_internal_id: u32,
    gallery: Gallery,
    #[serde(rename = "entries")]
    cards: Vec<PlayerCardState>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PlayerCardState {
    #[serde(rename = "deck_id")]
    manager_internal_id: u32,
    card_id: u16,
    config: PlayerCardConfig,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PlayerCardConfig {
    idolized: bool,
    level: u32,
    skill_level: u8,
    sis_slots: u8,
}

impl CardManager {
    pub fn new(card_factory: Arc<CardFactory>) -> Self {
        Self {
            card_factory,
            cards: HashMap::new(),
            next_manager_internal_id: 1,
            gallery: Arc::new(RwLock::new(Gallery::default())),
        }
    }

    pub fn gallery(&self) -> Arc<RwLock<Gallery>> {
        Arc::clone(&self.gallery)
    }

    pub fn set_gallery(&mut self, new_gallery: Gallery) {
        *self.gallery.write().unwrap() = new_gallery;
    }

    pub fn add_card(
        &mut self,
        card_id: u16,
        idolized: bool,
        skill_level: u8,
        level: Option<u32>,
        sis_slots: Option<u8>,
    ) -> Option<u32> {
        let card = self.card_factory.create_card(
            card_id,
            Arc::clone(&self.gallery),
            idolized,
            skill_level,
            level,
            sis_slots,
        )?;

        let manager_internal_id = self.next_manager_internal_id;
        self.cards.insert(
            manager_internal_id,
            PlayerCard {
                manager_internal_id,
                card,
            },
        );
        self.next_manager_internal_id += 1;
        Some(manager_internal_id)
    }

    pub fn get_card(&self, manager_internal_id: u32) -> Option<&Card> {
        self.cards
            .get(&manager_internal_id)
            .map(|entry| &entry.card)
    }

    pub fn remove_card(&mut self, manager_internal_id: u32) -> bool {
        self.cards.remove(&manager_internal_id).is_some()
    }

    pub fn modify_card(
        &mut self,
        manager_internal_id: u32,
        idolized: bool,
        skill_level: u8,
        level: Option<u32>,
        sis_slots: Option<u8>,
    ) -> bool {
        let card_id = if let Some(entry) = self.cards.get(&manager_internal_id) {
            entry.card.card_data.card_id
        } else {
            return false;
        };

        let new_card = self.card_factory.create_card(
            card_id,
            Arc::clone(&self.gallery),
            idolized,
            skill_level,
            level,
            sis_slots,
        );

        if let Some(new_card) = new_card {
            self.cards.insert(
                manager_internal_id,
                PlayerCard {
                    manager_internal_id,
                    card: new_card,
                },
            );
            true
        } else {
            false
        }
    }

    pub fn get_unassigned_cards(&self, assigned_card_ids: &HashSet<u32>) -> Vec<&Card> {
        self.cards
            .iter()
            .filter(|(manager_internal_id, _)| !assigned_card_ids.contains(manager_internal_id))
            .map(|(_, entry)| &entry.card)
            .collect()
    }

    pub fn cards(&self) -> &HashMap<u32, PlayerCard> {
        &self.cards
    }

    pub fn get_player_card(&self, manager_internal_id: u32) -> Option<&PlayerCard> {
        self.cards.get(&manager_internal_id)
    }

    pub fn save(&self, filepath: &str) -> Result<(), io::Error> {
        let state = CardManagerState {
            next_manager_internal_id: self.next_manager_internal_id,
            gallery: self.gallery.read().unwrap().clone(),
            cards: {
                let mut card_states: Vec<_> = self
                    .cards
                    .values()
                    .map(|pc| PlayerCardState {
                        manager_internal_id: pc.manager_internal_id,
                        card_id: pc.card.card_data.card_id,
                        config: PlayerCardConfig {
                            idolized: pc.card.idolized_status == "idolized",
                            level: pc.card.level,
                            skill_level: pc.card.skill_level(),
                            sis_slots: pc.card.current_sis_slots(),
                        },
                    })
                    .collect();
                card_states.sort_by_key(|s| s.manager_internal_id);
                card_states
            },
        };

        if let Some(dir) = std::path::Path::new(filepath).parent() {
            fs::create_dir_all(dir)?;
        }

        let file = fs::File::create(filepath)?;
        serde_json::to_writer_pretty(file, &state).map_err(io::Error::other)
    }

    pub fn load(&mut self, filepath: &str) -> Result<(), io::Error> {
        let file = fs::File::open(filepath)?;
        let state: CardManagerState = serde_json::from_reader(file).map_err(io::Error::other)?;

        self.delete();
        self.set_gallery(state.gallery);

        for card_data in state.cards {
            if let Some(card) = self.card_factory.create_card(
                card_data.card_id,
                Arc::clone(&self.gallery),
                card_data.config.idolized,
                card_data.config.skill_level,
                Some(card_data.config.level),
                Some(card_data.config.sis_slots),
            ) {
                let player_card = PlayerCard {
                    manager_internal_id: card_data.manager_internal_id,
                    card,
                };
                self.cards
                    .insert(player_card.manager_internal_id, player_card);
            } else {
                eprintln!("Could not create card with id {}", card_data.card_id);
            }
        }

        self.next_manager_internal_id = state.next_manager_internal_id;
        Ok(())
    }

    pub fn delete(&mut self) {
        self.cards.clear();
        *self.gallery.write().unwrap() = Gallery::default();
        self.next_manager_internal_id = 1;
    }
}

impl fmt::Display for CardManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.cards.is_empty() {
            return write!(f, "<CardManager (empty)>");
        }

        writeln!(f, "<CardManager ({} cards)>", self.cards.len())?;

        let mut sorted_cards: Vec<_> = self.cards.values().collect();
        sorted_cards.sort_by_key(|pc| pc.manager_internal_id);

        for (i, pc) in sorted_cards.iter().enumerate() {
            write!(f, "  - ID {}: {}", pc.manager_internal_id, pc.card)?;
            if i < sorted_cards.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
