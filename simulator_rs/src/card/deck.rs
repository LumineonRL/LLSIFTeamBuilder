use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use super::card_factory::CardFactory;
use super::core::Card;
use super::gallery::Gallery;

#[derive(Serialize, Deserialize)]
pub struct DeckEntryConfig {
    idolized: bool,
    level: u32,
    skill_level: u8,
    sis_slots: u8,
}

#[derive(Serialize, Deserialize)]
pub struct DeckEntrySerializable {
    deck_id: u32,
    card_id: u32,
    config: DeckEntryConfig,
}

pub struct DeckEntry {
    pub deck_id: u32,
    pub card: Card,
}

pub struct Deck {
    card_factory: Arc<CardFactory>,
    entries: HashMap<u32, DeckEntry>,
    next_deck_id: u32,
    gallery: Arc<RwLock<Gallery>>,
}

impl Deck {
    pub fn new(card_factory: Arc<CardFactory>) -> Self {
        Self {
            card_factory,
            entries: HashMap::new(),
            next_deck_id: 1,
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
        card_id: u32,
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

        let deck_id = self.next_deck_id;
        self.entries.insert(deck_id, DeckEntry { deck_id, card });
        self.next_deck_id += 1;
        Some(deck_id)
    }

    pub fn get_card(&self, deck_id: u32) -> Option<&Card> {
        self.entries.get(&deck_id).map(|entry| &entry.card)
    }

    pub fn remove_card(&mut self, deck_id: u32) -> bool {
        if self.entries.remove(&deck_id).is_none() {
            println!("Deck ID {deck_id} not found for removal.");
            return false;
        }
        true
    }

    pub fn modify_card(
        &mut self,
        deck_id: u32,
        idolized: bool,
        skill_level: u8,
        level: Option<u32>,
        sis_slots: Option<u8>,
    ) -> bool {
        let entry = match self.entries.get(&deck_id) {
            Some(entry) => entry,
            None => {
                println!("Deck ID {deck_id} not found for modification.");
                return false;
            }
        };

        let new_card = self.card_factory.create_card(
            entry.card.card_data.card_id,
            Arc::clone(&self.gallery),
            idolized,
            skill_level,
            level,
            sis_slots,
        );

        if let Some(new_card) = new_card {
            self.entries.insert(
                deck_id,
                DeckEntry {
                    deck_id,
                    card: new_card,
                },
            );
            true
        } else {
            println!("Failed to modify card with Deck ID {deck_id}. Re-creation failed.");
            false
        }
    }

    pub fn get_unassigned_cards(&self, assigned_deck_ids: &HashSet<u32>) -> Vec<&Card> {
        self.entries
            .iter()
            .filter(|(deck_id, _)| !assigned_deck_ids.contains(deck_id))
            .map(|(_, entry)| &entry.card)
            .collect()
    }

    pub fn to_serializable(&self) -> (Gallery, Vec<DeckEntrySerializable>) {
        let gallery = self.gallery.read().unwrap().clone();
        let entries = self
            .entries
            .values()
            .map(|entry| DeckEntrySerializable {
                deck_id: entry.deck_id,
                card_id: entry.card.card_data.card_id,
                config: DeckEntryConfig {
                    idolized: entry.card.idolized_status == "idolized",
                    level: entry.card.level,
                    skill_level: entry.card.skill_level(),
                    sis_slots: entry.card.current_sis_slots(),
                },
            })
            .collect();
        (gallery, entries)
    }

    pub fn load_from_serializable(
        &mut self,
        gallery: Gallery,
        entries: Vec<DeckEntrySerializable>,
        next_deck_id: u32,
    ) {
        self.delete_deck();
        self.set_gallery(gallery);

        for entry_data in entries {
            let card = self.card_factory.create_card(
                entry_data.card_id,
                Arc::clone(&self.gallery),
                entry_data.config.idolized,
                entry_data.config.skill_level,
                Some(entry_data.config.level),
                Some(entry_data.config.sis_slots),
            );

            if let Some(card) = card {
                self.entries.insert(
                    entry_data.deck_id,
                    DeckEntry {
                        deck_id: entry_data.deck_id,
                        card,
                    },
                );
            } else {
                println!(
                    "Skipping card in deck file due to creation failure: card_id {}",
                    entry_data.card_id
                );
            }
        }
        self.next_deck_id = next_deck_id;
    }

    pub fn delete_deck(&mut self) {
        self.entries.clear();
        *self.gallery.write().unwrap() = Gallery::default();
        self.next_deck_id = 1;
    }
}
