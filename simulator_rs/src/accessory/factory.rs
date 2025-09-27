use crate::accessory::accessory_data::{AccessoryData, AccessoryDataRaw};
use crate::accessory::core::Accessory;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AccessoryFactory {
    accessory_data_map: HashMap<u32, Arc<AccessoryData>>,
}

impl AccessoryFactory {
    pub fn new(accessories_json_path: &str) -> Result<Self, String> {
        let file = File::open(accessories_json_path)
            .map_err(|e| format!("Failed to open accessory JSON file: {e}"))?;
        let reader = BufReader::new(file);
        let raw_data: HashMap<String, AccessoryDataRaw> = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse accessory JSON: {e}"))?;

        let accessory_data_map = raw_data
            .into_iter()
            .filter_map(|(id_str, raw_data)| {
                id_str.parse::<u32>().ok().map(|id| {
                    let data = AccessoryData {
                        accessory_id: id,
                        name: raw_data.name,
                        character: raw_data.character,
                        card_id: raw_data.card_id,
                        stats: raw_data.stats,
                        skill: raw_data.skill,
                    };
                    (id, Arc::new(data))
                })
            })
            .collect();

        Ok(Self { accessory_data_map })
    }

    pub fn create_accessory(&self, accessory_id: u32, skill_level: u8) -> Option<Accessory> {
        self.accessory_data_map
            .get(&accessory_id)
            .and_then(|data| Accessory::new(data.clone(), skill_level).ok())
    }
}
