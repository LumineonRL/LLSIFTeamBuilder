use crate::sis::core::Sis;
use crate::sis::data::SisData;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SisFactory {
    sis_data_map: HashMap<u32, Arc<SisData>>,
}

impl SisFactory {
    pub fn new(sis_json_path: &str) -> Result<Self, String> {
        let file =
            File::open(sis_json_path).map_err(|e| format!("Failed to open SIS JSON file: {e}"))?;
        let reader = BufReader::new(file);
        let raw_data: Vec<SisData> = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse SIS JSON: {e}"))?;

        let sis_data_map = raw_data
            .into_iter()
            .map(|data| (data.id, Arc::new(data)))
            .collect();

        Ok(Self { sis_data_map })
    }

    pub fn create_sis(&self, sis_id: u32) -> Option<Sis> {
        self.sis_data_map
            .get(&sis_id)
            .map(|data| Sis::new(data.clone()))
    }
}
