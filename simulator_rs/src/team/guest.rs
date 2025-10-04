use crate::core::leader_skill::LeaderSkill;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Extra {
    leader_extra_attribute: Option<String>,
    leader_extra_target: Option<String>,
    leader_extra_value: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GuestData {
    pub leader_skill_id: u32,
    pub leader_attribute: Option<String>,
    pub leader_secondary_attribute: Option<String>,
    pub leader_value: Option<f32>,
    extra: Extra,
}

impl Display for GuestData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let leader_attribute = self
            .leader_attribute
            .as_ref()
            .map_or("None".to_string(), |s| format!("'{}'", s));
        let leader_secondary_attribute = self
            .leader_secondary_attribute
            .as_ref()
            .map_or("None".to_string(), |s| format!("'{}'", s));
        let leader_value = self
            .leader_value
            .map_or("None".to_string(), |v| v.to_string());
        let leader_extra_attribute = self
            .extra
            .leader_extra_attribute
            .as_ref()
            .map_or("None".to_string(), |s| format!("'{}'", s));
        let leader_extra_target = self
            .extra
            .leader_extra_target
            .as_ref()
            .map_or("None".to_string(), |s| format!("'{}'", s));
        let leader_extra_value = self
            .extra
            .leader_extra_value
            .map_or("None".to_string(), |v| v.to_string());

        write!(
            f,
            "GuestData(leader_skill_id={}, leader_attribute={}, leader_secondary_attribute={}, leader_value={}, leader_extra_attribute={}, leader_extra_target={}, leader_extra_value={})",
            self.leader_skill_id,
            leader_attribute,
            leader_secondary_attribute,
            leader_value,
            leader_extra_attribute,
            leader_extra_target,
            leader_extra_value
        )
    }
}

#[derive(Debug, Default)]
pub struct Guest {
    all_guests: HashMap<u32, GuestData>,
    current_guest: Option<GuestData>,
}

impl Guest {
    pub fn new(unique_skills_path: &str) -> Result<Self, String> {
        let all_guests = Self::load_and_index_guests(unique_skills_path)?;
        Ok(Self {
            all_guests,
            current_guest: None,
        })
    }

    fn load_and_index_guests(filepath: &str) -> Result<HashMap<u32, GuestData>, String> {
        let file =
            File::open(filepath).map_err(|e| format!("Failed to open guest data file: {e}"))?;
        let reader = BufReader::new(file);
        let guests: Vec<GuestData> = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse guest data: {e}"))?;

        let indexed_guests = guests.into_iter().map(|g| (g.leader_skill_id, g)).collect();
        Ok(indexed_guests)
    }

    pub fn set_guest(&mut self, leader_skill_id: u32) -> bool {
        if let Some(guest_data) = self.all_guests.get(&leader_skill_id) {
            self.current_guest = Some(guest_data.clone());
            true
        } else {
            self.current_guest = None;
            false
        }
    }

    pub fn leader_skill(&self) -> Option<LeaderSkill> {
        self.current_guest.as_ref().map(|guest_data| {
            LeaderSkill::new(
                guest_data.leader_attribute.clone(),
                guest_data.leader_secondary_attribute.clone(),
                guest_data.leader_value,
                guest_data.extra.leader_extra_attribute.clone(),
                guest_data.extra.leader_extra_target.clone(),
                guest_data.extra.leader_extra_value,
            )
        })
    }

    pub fn all_guests(&self) -> &HashMap<u32, GuestData> {
        &self.all_guests
    }

    pub fn current_guest(&self) -> Option<&GuestData> {
        self.current_guest.as_ref()
    }
}

impl Display for Guest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(guest_data) = &self.current_guest {
            writeln!(f, "--- Current Guest Details ---")?;
            writeln!(f, "Leader Skill Id: {}", guest_data.leader_skill_id)?;
            writeln!(
                f,
                "Leader Attribute: {}",
                guest_data.leader_attribute.as_deref().unwrap_or("N/A")
            )?;
            writeln!(
                f,
                "Leader Secondary Attribute: {}",
                guest_data
                    .leader_secondary_attribute
                    .as_deref()
                    .unwrap_or("N/A")
            )?;
            writeln!(
                f,
                "Leader Value: {}",
                guest_data
                    .leader_value
                    .map_or("N/A".to_string(), |v| v.to_string())
            )?;
            writeln!(
                f,
                "Leader Extra Attribute: {}",
                guest_data
                    .extra
                    .leader_extra_attribute
                    .as_deref()
                    .unwrap_or("N/A")
            )?;
            writeln!(
                f,
                "Leader Extra Target: {}",
                guest_data
                    .extra
                    .leader_extra_target
                    .as_deref()
                    .unwrap_or("N/A")
            )?;
            write!(
                f,
                "Leader Extra Value: {}",
                guest_data
                    .extra
                    .leader_extra_value
                    .map_or("N/A".to_string(), |v| v.to_string())
            )?;
            write!(f, "\n---------------------------")
        } else {
            write!(f, "<Guest active_id=None>")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_json(dir: &tempfile::TempDir) -> String {
        let path = dir.path().join("test_guests.json");
        let mut file = File::create(&path).unwrap();
        writeln!(
            file,
            r#"[
                {{
                    "leader_skill_id": 1,
                    "leader_attribute": "Smile",
                    "leader_secondary_attribute": null,
                    "leader_value": 0.09,
                    "extra": {{
                        "leader_extra_attribute": "Smile",
                        "leader_extra_target": "μ's",
                        "leader_extra_value": 0.03
                    }}
                }},
                {{
                    "leader_skill_id": 2,
                    "leader_attribute": "Pure",
                    "leader_secondary_attribute": null,
                    "leader_value": 0.09,
                    "extra": {{
                        "leader_extra_attribute": null,
                        "leader_extra_target": null,
                        "leader_extra_value": null
                    }}
                }}
            ]"#
        )
        .unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_load_and_index_guests() {
        let dir = tempdir().unwrap();
        let path = create_test_json(&dir);
        let guest_manager = Guest::new(&path).unwrap();

        assert_eq!(guest_manager.all_guests().len(), 2);
        assert!(guest_manager.all_guests().contains_key(&1));
        assert!(guest_manager.all_guests().contains_key(&2));
    }

    #[test]
    fn test_set_guest() {
        let dir = tempdir().unwrap();
        let path = create_test_json(&dir);
        let mut guest_manager = Guest::new(&path).unwrap();

        assert!(guest_manager.set_guest(1));
        assert!(guest_manager.current_guest.is_some());
        assert_eq!(
            guest_manager
                .current_guest
                .as_ref()
                .unwrap()
                .leader_skill_id,
            1
        );

        assert!(!guest_manager.set_guest(3));
        assert!(guest_manager.current_guest.is_none());
    }

    #[test]
    fn test_leader_skill() {
        let dir = tempdir().unwrap();
        let path = create_test_json(&dir);
        let mut guest_manager = Guest::new(&path).unwrap();

        guest_manager.set_guest(1);
        let leader_skill = guest_manager.leader_skill().unwrap();

        assert_eq!(leader_skill.attribute, Some("Smile".to_string()));
        assert_eq!(leader_skill.value, Some(0.09));
        assert_eq!(leader_skill.extra_attribute(), Some(&"Smile".to_string()));
        assert_eq!(leader_skill.extra_target(), Some(&"μ's".to_string()));
        assert_eq!(leader_skill.extra_value(), 0.03);
    }

    #[test]
    fn test_display_guest() {
        let dir = tempdir().unwrap();
        let path = create_test_json(&dir);
        let mut guest_manager = Guest::new(&path).unwrap();
        guest_manager.set_guest(1);

        let expected_display = r#"--- Current Guest Details ---
Leader Skill Id: 1
Leader Attribute: Smile
Leader Secondary Attribute: N/A
Leader Value: 0.09
Leader Extra Attribute: Smile
Leader Extra Target: μ's
Leader Extra Value: 0.03
---------------------------"#;
        assert_eq!(guest_manager.to_string(), expected_display);
    }

    #[test]
    fn test_display_no_guest() {
        let dir = tempdir().unwrap();
        let path = create_test_json(&dir);
        let guest_manager = Guest::new(&path).unwrap();
        assert_eq!(guest_manager.to_string(), "<Guest active_id=None>");
    }
}
