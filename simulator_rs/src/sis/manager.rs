use super::SIS;
use super::sis_factory::SISFactory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PlayerSIS {
    pub manager_internal_id: u32,
    pub sis: Arc<SIS>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SISSaveData {
    manager_internal_id: u32,
    sid: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct SISManagerSaveState {
    next_manager_internal_id: u32,
    skills: Vec<SISSaveData>,
}

pub struct SISManager {
    factory: Arc<SISFactory>,
    skills: HashMap<u32, PlayerSIS>,
    next_manager_internal_id: u32,
}

impl SISManager {
    pub fn new(factory: Arc<SISFactory>) -> Self {
        Self {
            factory,
            skills: HashMap::new(),
            next_manager_internal_id: 1,
        }
    }

    pub fn add_sis(&mut self, sid: u32) -> Result<u32, String> {
        match self.factory.create_sis(sid) {
            Some(sis) => {
                let manager_id = self.next_manager_internal_id;
                self.skills.insert(
                    manager_id,
                    PlayerSIS {
                        manager_internal_id: manager_id,
                        sis,
                    },
                );
                self.next_manager_internal_id += 1;
                Ok(manager_id)
            }
            None => Err(format!("SIS with SID {sid} not found in factory")),
        }
    }

    pub fn get_sis(&self, manager_internal_id: u32) -> Option<Arc<SIS>> {
        self.skills
            .get(&manager_internal_id)
            .map(|ps| ps.sis.clone())
    }

    pub fn get_player_sis(&self, manager_internal_id: u32) -> Option<PlayerSIS> {
        self.skills.get(&manager_internal_id).cloned()
    }

    pub fn remove_sis(&mut self, manager_internal_id: u32) -> bool {
        self.skills.remove(&manager_internal_id).is_some()
    }

    pub fn save(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(p) = Path::new(filepath).parent() {
            fs::create_dir_all(p)?;
        }
        let mut file = File::create(filepath)?;

        let mut skills_to_save: Vec<SISSaveData> = self
            .skills
            .values()
            .map(|ps| SISSaveData {
                manager_internal_id: ps.manager_internal_id,
                sid: ps.sis.id,
            })
            .collect();
        skills_to_save.sort_by_key(|s| s.manager_internal_id);

        let state = SISManagerSaveState {
            next_manager_internal_id: self.next_manager_internal_id,
            skills: skills_to_save,
        };

        let json_string = serde_json::to_string_pretty(&state)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }

    pub fn load(&mut self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let state: SISManagerSaveState = serde_json::from_reader(reader)?;

        self.skills.clear();

        for item_data in state.skills {
            if let Some(sis) = self.factory.create_sis(item_data.sid) {
                self.skills.insert(
                    item_data.manager_internal_id,
                    PlayerSIS {
                        manager_internal_id: item_data.manager_internal_id,
                        sis,
                    },
                );
            }
        }

        self.next_manager_internal_id = state.next_manager_internal_id;
        Ok(())
    }
}

impl fmt::Display for SISManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.skills.is_empty() {
            return write!(f, "<SISManager (empty)>");
        }

        writeln!(f, "<SISManager ({} skills)>", self.skills.len())?;

        let mut sorted_skills: Vec<&PlayerSIS> = self.skills.values().collect();
        sorted_skills.sort_by_key(|ps| ps.manager_internal_id);

        for ps in sorted_skills {
            writeln!(
                f,
                "  - ID {}: {} (SID: {})",
                ps.manager_internal_id, ps.sis.name, ps.sis.id
            )?;
        }
        Ok(())
    }
}
