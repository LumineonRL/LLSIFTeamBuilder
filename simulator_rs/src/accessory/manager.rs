use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;

use crate::accessory::core::Accessory;
use crate::accessory::factory::AccessoryFactory;

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerAccessory {
    pub manager_internal_id: u32,
    pub accessory: Accessory,
}

#[derive(Debug, Clone)]
pub struct AccessoryManager {
    factory: AccessoryFactory,
    accessories: HashMap<u32, PlayerAccessory>,
    next_manager_internal_id: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct AccessoryManagerState {
    next_manager_internal_id: u32,
    accessories: Vec<PlayerAccessoryState>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PlayerAccessoryState {
    manager_internal_id: u32,
    accessory_id: u16,
    skill_level: u8,
}

impl AccessoryManager {
    pub fn new(factory: AccessoryFactory) -> Self {
        Self {
            factory,
            accessories: HashMap::new(),
            next_manager_internal_id: 1,
        }
    }

    pub fn accessories(&self) -> &HashMap<u32, PlayerAccessory> {
        &self.accessories
    }

    pub fn add_accessory(&mut self, accessory_id: u16, skill_level: u8) -> Option<u32> {
        let accessory = self.factory.create_accessory(accessory_id, skill_level)?;

        let manager_id = self.next_manager_internal_id;
        self.accessories.insert(
            manager_id,
            PlayerAccessory {
                manager_internal_id: manager_id,
                accessory,
            },
        );
        self.next_manager_internal_id += 1;
        Some(manager_id)
    }

    pub fn get_accessory(&self, manager_internal_id: u32) -> Option<&Accessory> {
        self.accessories
            .get(&manager_internal_id)
            .map(|pa| &pa.accessory)
    }

    pub fn remove_accessory(&mut self, manager_internal_id: u32) -> bool {
        if self.accessories.contains_key(&manager_internal_id) {
            self.accessories.remove(&manager_internal_id);
            true
        } else {
            false
        }
    }

    pub fn modify_accessory(&mut self, manager_internal_id: u32, skill_level: Option<u8>) -> bool {
        if let Some(player_accessory) = self.accessories.get_mut(&manager_internal_id) {
            if let Some(level) = skill_level
                && player_accessory.accessory.set_skill_level(level).is_err()
            {
                return false;
            }
            true
        } else {
            false
        }
    }

    pub fn get_unassigned_accessories(
        &self,
        assigned_accessory_ids: &HashSet<u32>,
    ) -> Vec<&PlayerAccessory> {
        self.accessories
            .iter()
            .filter(|(manager_id, _)| !assigned_accessory_ids.contains(manager_id))
            .map(|(_, pa)| pa)
            .collect()
    }

    pub fn get_player_accessory(&self, manager_internal_id: u32) -> Option<&PlayerAccessory> {
        self.accessories.get(&manager_internal_id)
    }

    pub fn delete(&mut self) {
        self.accessories.clear();
        self.next_manager_internal_id = 1;
    }

    pub fn save(&self, filepath: &str) -> Result<(), io::Error> {
        let mut sorted_accessories: Vec<_> = self.accessories.values().collect();
        sorted_accessories.sort_by_key(|pa| pa.manager_internal_id);

        let state = AccessoryManagerState {
            next_manager_internal_id: self.next_manager_internal_id,
            accessories: sorted_accessories
                .iter()
                .map(|pa| PlayerAccessoryState {
                    manager_internal_id: pa.manager_internal_id,
                    accessory_id: pa.accessory.data.accessory_id,
                    skill_level: pa.accessory.skill_level,
                })
                .collect(),
        };

        if let Some(dir) = std::path::Path::new(filepath).parent() {
            fs::create_dir_all(dir)?;
        }

        let file = fs::File::create(filepath)?;
        serde_json::to_writer_pretty(file, &state).map_err(io::Error::other)
    }

    pub fn load(&mut self, filepath: &str) -> Result<(), io::Error> {
        let file = fs::File::open(filepath)?;
        let state: AccessoryManagerState =
            serde_json::from_reader(file).map_err(io::Error::other)?;

        self.delete();

        for item_data in state.accessories {
            if let Some(accessory) = self
                .factory
                .create_accessory(item_data.accessory_id, item_data.skill_level)
            {
                let player_acc = PlayerAccessory {
                    manager_internal_id: item_data.manager_internal_id,
                    accessory,
                };
                self.accessories
                    .insert(player_acc.manager_internal_id, player_acc);
            } else {
                eprintln!(
                    "Could not create accessory with id {}",
                    item_data.accessory_id
                );
            }
        }

        self.next_manager_internal_id = state.next_manager_internal_id;
        Ok(())
    }
}

impl fmt::Display for AccessoryManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.accessories.is_empty() {
            return write!(f, "<AccessoryManager (empty)>");
        }

        let header = format!(
            "<AccessoryManager ({} accessories)>",
            self.accessories.len()
        );

        let mut sorted_accessories: Vec<_> = self.accessories.values().collect();
        sorted_accessories.sort_by_key(|pa| pa.manager_internal_id);

        let items: Vec<String> = sorted_accessories
            .iter()
            .map(|pa| format!("  - ID {}: {}", pa.manager_internal_id, pa.accessory))
            .collect();

        write!(f, "{}\n{}", header, items.join("\n"))
    }
}
