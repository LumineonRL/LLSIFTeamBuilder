use super::SIS;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

pub struct SISFactory {
    sis_data: HashMap<u32, Arc<SIS>>,
}

impl SISFactory {
    pub fn new(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let sises: Vec<SIS> = serde_json::from_reader(reader)?;
        let sis_data = sises.into_iter().map(|c| (c.id, Arc::new(c))).collect();
        Ok(Self { sis_data })
    }

    pub fn create_sis(&self, sid: u32) -> Option<Arc<SIS>> {
        self.sis_data.get(&sid).cloned()
    }
}
