use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;

use crate::sis::core::Sis;
use crate::sis::factory::SisFactory;

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerSis {
    pub manager_internal_id: u32,
    pub sis: Sis,
}

#[derive(Debug, Clone)]
pub struct SisManager {
    factory: SisFactory,
    skills: HashMap<u32, PlayerSis>,
    next_manager_internal_id: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct SisManagerState {
    next_manager_internal_id: u32,
    skills: Vec<PlayerSisState>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PlayerSisState {
    manager_internal_id: u32,
    sid: u32,
}

impl SisManager {
    pub fn new(factory: SisFactory) -> Self {
        Self {
            factory,
            skills: HashMap::new(),
            next_manager_internal_id: 1,
        }
    }

    pub fn skills(&self) -> &HashMap<u32, PlayerSis> {
        &self.skills
    }

    pub fn add_sis(&mut self, sid: u32) -> Option<u32> {
        let sis = self.factory.create_sis(sid)?;

        let manager_id = self.next_manager_internal_id;
        self.skills.insert(
            manager_id,
            PlayerSis {
                manager_internal_id: manager_id,
                sis,
            },
        );
        self.next_manager_internal_id += 1;
        Some(manager_id)
    }

    pub fn get_sis(&self, manager_internal_id: u32) -> Option<&Sis> {
        self.skills.get(&manager_internal_id).map(|ps| &ps.sis)
    }

    pub fn remove_sis(&mut self, manager_internal_id: u32) -> bool {
        if self.skills.contains_key(&manager_internal_id) {
            self.skills.remove(&manager_internal_id);
            true
        } else {
            false
        }
    }

    pub fn get_unassigned_sis(&self, assigned_sis_ids: &HashSet<u32>) -> Vec<&PlayerSis> {
        self.skills
            .iter()
            .filter(|(manager_id, _)| !assigned_sis_ids.contains(manager_id))
            .map(|(_, ps)| ps)
            .collect()
    }

    pub fn get_player_sis(&self, manager_internal_id: u32) -> Option<&PlayerSis> {
        self.skills.get(&manager_internal_id)
    }

    pub fn delete(&mut self) {
        self.skills.clear();
        self.next_manager_internal_id = 1;
    }

    pub fn save(&self, filepath: &str) -> Result<(), io::Error> {
        let state = SisManagerState {
            next_manager_internal_id: self.next_manager_internal_id,
            skills: self
                .skills
                .values()
                .map(|ps| PlayerSisState {
                    manager_internal_id: ps.manager_internal_id,
                    sid: ps.sis.id(),
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
        let state: SisManagerState = serde_json::from_reader(file).map_err(io::Error::other)?;

        self.delete();

        for item_data in state.skills {
            if let Some(sis) = self.factory.create_sis(item_data.sid) {
                let player_sis = PlayerSis {
                    manager_internal_id: item_data.manager_internal_id,
                    sis,
                };
                self.skills
                    .insert(player_sis.manager_internal_id, player_sis);
            } else {
                eprintln!("Could not create SIS with id {}", item_data.sid);
            }
        }

        self.next_manager_internal_id = state.next_manager_internal_id;
        Ok(())
    }
}

impl fmt::Display for SisManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.skills.is_empty() {
            return write!(f, "<SisManager (empty)>");
        }

        writeln!(f, "<SisManager ({} skills)>", self.skills.len())?;

        let mut sorted_skills: Vec<_> = self.skills.values().collect();
        sorted_skills.sort_by_key(|ps| ps.manager_internal_id);

        for (i, ps) in sorted_skills.iter().enumerate() {
            write!(
                f,
                "  - ID {}: {} (SID: {})",
                ps.manager_internal_id,
                ps.sis.name(),
                ps.sis.id()
            )?;
            if i < sorted_skills.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
