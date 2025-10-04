use crate::accessory::manager::AccessoryManager;
use crate::card::manager::CardManager;
use crate::core::leader_skill::LeaderSkill;
use crate::sis::SIS;
use crate::sis::manager::SISManager;
use crate::team::guest::Guest;
use crate::team::team_slot::TeamSlot;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};

const NUM_SLOTS: usize = 9;
const CENTER_SLOT_NUMBER: usize = 5;

pub struct Team<'a> {
    card_manager: &'a CardManager,
    accessory_manager: &'a AccessoryManager,
    sis_manager: &'a SISManager,
    guest_manager: Option<&'a Guest>,
    pub slots: Vec<TeamSlot>,
    assigned_deck_ids: HashSet<u32>,
    assigned_accessory_ids: HashSet<u32>,
    assigned_sis_ids: HashSet<u32>,
    pub total_team_smile: u32,
    pub total_team_pure: u32,
    pub total_team_cool: u32,
    year_group_mapping: HashMap<String, HashSet<String>>,
    group_member_mapping: HashMap<String, HashSet<String>>,
    additional_skill_map: HashMap<String, HashSet<String>>,
}

impl<'a> Team<'a> {
    pub fn new(
        card_manager: &'a CardManager,
        accessory_manager: &'a AccessoryManager,
        sis_manager: &'a SISManager,
        guest_manager: Option<&'a Guest>,
    ) -> Result<Self, String> {
        Ok(Self {
            card_manager,
            accessory_manager,
            sis_manager,
            guest_manager,
            slots: (0..NUM_SLOTS).map(|_| TeamSlot::new()).collect(),
            assigned_deck_ids: HashSet::new(),
            assigned_accessory_ids: HashSet::new(),
            assigned_sis_ids: HashSet::new(),
            total_team_smile: 0,
            total_team_pure: 0,
            total_team_cool: 0,
            year_group_mapping: Self::load_json_mapping("data/year_group_mapping.json")
                .map_err(|e| e.to_string())?,
            group_member_mapping: Self::load_json_mapping("data/group_member_map.json")
                .map_err(|e| e.to_string())?,
            additional_skill_map: Self::load_json_mapping("data/additional_leader_skill_map.json")
                .map_err(|e| e.to_string())?,
        })
    }

    fn load_json_mapping(
        filepath: &str,
    ) -> Result<HashMap<String, HashSet<String>>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(filepath)?;
        let reader = std::io::BufReader::new(file);
        let mapping: HashMap<String, Vec<String>> = serde_json::from_reader(reader)?;
        Ok(mapping
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect())
    }

    pub fn center_slot(&self) -> &TeamSlot {
        &self.slots[CENTER_SLOT_NUMBER - 1]
    }
    fn _get_slot(&self, i: usize) -> Result<&TeamSlot, String> {
        self.slots
            .get(i - 1)
            .ok_or_else(|| "Invalid slot".to_string())
    }
    fn _get_slot_mut(&mut self, i: usize) -> Result<&mut TeamSlot, String> {
        self.slots
            .get_mut(i - 1)
            .ok_or_else(|| "Invalid slot".to_string())
    }

    pub fn equip_card_in_slot(&mut self, slot_number: usize, deck_id: u32) -> Result<(), String> {
        if self.assigned_deck_ids.contains(&deck_id) {
            return Err("Card already assigned".to_string());
        }
        let card_entry = self
            .card_manager
            .get_player_card(deck_id)
            .cloned()
            .ok_or("Card not found")?;
        self.clear_slot(slot_number)?;
        self._get_slot_mut(slot_number)?.equip_card(card_entry);
        self.assigned_deck_ids.insert(deck_id);
        self.calculate_team_stats();
        Ok(())
    }

    pub fn equip_accessory_in_slot(
        &mut self,
        slot_number: usize,
        manager_id: u32,
    ) -> Result<(), String> {
        let card = self
            ._get_slot(slot_number)?
            .card()
            .cloned()
            .ok_or("No card in slot")?;
        if self.assigned_accessory_ids.contains(&manager_id) {
            return Err("Accessory already assigned".to_string());
        }
        let acc_entry = self
            .accessory_manager
            .get_player_accessory(manager_id)
            .cloned()
            .ok_or("Accessory not found")?;
        if let Some(id_str) = &acc_entry.accessory.data.card_id
            && !id_str.is_empty()
            && id_str.parse::<u16>().ok() != Some(card.card_data.card_id)
        {
            return Err("Accessory-Card ID mismatch".to_string());
        }
        if let Some(existing) = self._get_slot_mut(slot_number)?.accessory_entry.take() {
            self.assigned_accessory_ids
                .remove(&existing.manager_internal_id);
        }
        self._get_slot_mut(slot_number)?
            .equip_accessory(acc_entry)
            .map_err(|e| e.to_string())?;
        self.assigned_accessory_ids.insert(manager_id);
        self.calculate_team_stats();
        Ok(())
    }

    pub fn equip_sis_in_slot(&mut self, slot_number: usize, manager_id: u32) -> Result<(), String> {
        let slot_clone = self._get_slot(slot_number)?.clone();
        if self.assigned_sis_ids.contains(&manager_id) {
            return Err("SIS already assigned".to_string());
        }
        let sis_entry = self
            .sis_manager
            .get_player_sis(manager_id)
            .ok_or("SIS not found")?;
        if slot_clone
            .sis_entries
            .iter()
            .any(|e| e.sis.id == sis_entry.sis.id)
        {
            return Err("Same SIS ID already in slot".to_string());
        }
        self.check_sis_equip_restriction(&slot_clone, &sis_entry.sis)?;
        self._get_slot_mut(slot_number)?
            .equip_sis(sis_entry)
            .map_err(|e| e.to_string())?;
        self.assigned_sis_ids.insert(manager_id);
        self.calculate_team_stats();
        Ok(())
    }

    pub fn check_sis_equip_restriction(&self, slot: &TeamSlot, sis: &SIS) -> Result<(), String> {
        let card = slot.card().ok_or("No card in slot")?;
        if sis.slots as u8 > slot.available_sis_slots() {
            return Err("Not enough SIS slots".to_string());
        }
        let r = &sis.equip_restriction;
        if r.is_empty() {
            return Ok(());
        }
        match r.as_str() {
            "Smile" | "Pure" | "Cool" if *r != card.card_data.attribute => {
                Err("Attribute mismatch".to_string())
            }
            _ if self.year_group_mapping.contains_key(r)
                && !self.year_group_mapping[r].contains(&card.card_data.character) =>
            {
                Err("Year group mismatch".to_string())
            }
            _ if !self.year_group_mapping.contains_key(r) && r != &card.card_data.character => {
                Err("Character mismatch".to_string())
            }
            _ => Ok(()),
        }
    }

    pub fn clear_slot(&mut self, slot_number: usize) -> Result<(), String> {
        let slot_clone = self._get_slot(slot_number)?.clone();
        if let Some(c) = slot_clone.card_entry {
            self.assigned_deck_ids.remove(&c.manager_internal_id);
        }
        if let Some(a) = slot_clone.accessory_entry {
            self.assigned_accessory_ids.remove(&a.manager_internal_id);
        }
        for s in slot_clone.sis_entries {
            self.assigned_sis_ids.remove(&s.manager_internal_id);
        }
        self._get_slot_mut(slot_number)?.clear();
        self.calculate_team_stats();
        Ok(())
    }

    pub fn calculate_team_stats(&mut self) {
        let boosts = self._calculate_all_percent_boosts();
        let center_ls = self.center_slot().card().map(|c| c.leader_skill.clone());
        let guest_ls = self.guest_manager.and_then(|g| g.leader_skill());

        let additional_skill_map = &self.additional_skill_map;

        for slot in &mut self.slots {
            let (mut s, mut p, mut c) = if let Some(card) = slot.card() {
                let st = card.stats();
                (st.smile, st.pure, st.cool)
            } else {
                slot.clear();
                continue;
            };
            if let Some(acc) = slot.accessory() {
                s += acc.stats.smile;
                p += acc.stats.pure;
                c += acc.stats.cool;
            }

            s = (s as f32 * (1.0 + boosts.get("Smile").unwrap_or(&0.0))).ceil() as u32;
            p = (p as f32 * (1.0 + boosts.get("Pure").unwrap_or(&0.0))).ceil() as u32;
            c = (c as f32 * (1.0 + boosts.get("Cool").unwrap_or(&0.0))).ceil() as u32;

            let mut self_b = HashMap::new();
            for se in &slot.sis_entries {
                if se.sis.effect == "self percent boost" {
                    *self_b.entry(se.sis.attribute.clone()).or_insert(0.0) += se.sis.value;
                }
            }
            s = (s as f32 * (1.0 + self_b.get("Smile").unwrap_or(&0.0))).ceil() as u32;
            p = (p as f32 * (1.0 + self_b.get("Pure").unwrap_or(&0.0))).ceil() as u32;
            c = (c as f32 * (1.0 + self_b.get("Cool").unwrap_or(&0.0))).ceil() as u32;

            for se in &slot.sis_entries {
                if se.sis.effect == "self flat boost" {
                    match se.sis.attribute.as_str() {
                        "Smile" => s += se.sis.value as u32,
                        "Pure" => p += se.sis.value as u32,
                        "Cool" => c += se.sis.value as u32,
                        _ => {}
                    }
                }
            }

            slot.total_smile = s;
            slot.total_pure = p;
            slot.total_cool = c;
            let center_b = _calculate_leader_skill_bonus(center_ls.as_ref(), slot);
            let guest_b = _calculate_leader_skill_bonus(guest_ls.as_ref(), slot);
            let center_e = center_ls.as_ref().map_or(HashMap::new(), |ls| {
                _calculate_extra_skill_bonus(ls, slot, additional_skill_map)
            });
            let guest_e = guest_ls.as_ref().map_or(HashMap::new(), |ls| {
                _calculate_extra_skill_bonus(ls, slot, additional_skill_map)
            });
            s += center_b.get("Smile").unwrap_or(&0)
                + guest_b.get("Smile").unwrap_or(&0)
                + center_e.get("Smile").unwrap_or(&0)
                + guest_e.get("Smile").unwrap_or(&0);
            p += center_b.get("Pure").unwrap_or(&0)
                + guest_b.get("Pure").unwrap_or(&0)
                + center_e.get("Pure").unwrap_or(&0)
                + guest_e.get("Pure").unwrap_or(&0);
            c += center_b.get("Cool").unwrap_or(&0)
                + guest_b.get("Cool").unwrap_or(&0)
                + center_e.get("Cool").unwrap_or(&0)
                + guest_e.get("Cool").unwrap_or(&0);
            slot.total_smile = s;
            slot.total_pure = p;
            slot.total_cool = c;
        }
        self.total_team_smile = self.slots.iter().map(|s| s.total_smile).sum();
        self.total_team_pure = self.slots.iter().map(|s| s.total_pure).sum();
        self.total_team_cool = self.slots.iter().map(|s| s.total_cool).sum();
    }

    fn _is_nonet_active(&self, group_name: &str) -> bool {
        let team_chars: HashSet<String> = self
            .slots
            .iter()
            .filter_map(|s| s.card().map(|c| c.card_data.character.clone()))
            .collect();
        team_chars.len() == 9
            && self
                .group_member_mapping
                .get(group_name)
                .is_some_and(|g| team_chars.is_subset(g))
    }

    fn _calculate_all_percent_boosts(&self) -> HashMap<String, f32> {
        let mut boosts = HashMap::new();
        let mut nonets = HashMap::new();
        for s in &self.slots {
            if s.card().is_none() {
                continue;
            }
            for se in &s.sis_entries {
                if se.sis.effect == "all percent boost" {
                    if !se.sis.group.is_empty() {
                        let is_active = *nonets
                            .entry(se.sis.group.clone())
                            .or_insert_with(|| self._is_nonet_active(&se.sis.group));
                        if is_active {
                            *boosts.entry(se.sis.attribute.clone()).or_insert(0.0) += se.sis.value;
                        }
                    } else {
                        *boosts.entry(se.sis.attribute.clone()).or_insert(0.0) += se.sis.value;
                    }
                }
            }
        }
        boosts
    }
}

fn _calculate_leader_skill_bonus(ls: Option<&LeaderSkill>, ts: &TeamSlot) -> HashMap<String, u32> {
    let mut bonuses = HashMap::new();
    if let Some(ls) = ls
        && let Some(p_attr) = &ls.attribute
    {
        let val = ls.value.unwrap_or(0.0);
        let src_stat = if let Some(s_attr) = &ls.secondary_attribute {
            match s_attr.as_str() {
                "Smile" => ts.total_smile,
                "Pure" => ts.total_pure,
                "Cool" => ts.total_cool,
                _ => 0,
            }
        } else {
            match p_attr.as_str() {
                "Smile" => ts.total_smile,
                "Pure" => ts.total_pure,
                "Cool" => ts.total_cool,
                _ => 0,
            }
        };
        bonuses.insert(p_attr.clone(), (src_stat as f32 * val).ceil() as u32);
    }
    bonuses
}

impl<'a> Display for Team<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- Team Configuration ---")?;

        let guest_line = if let Some(guest_manager) = self.guest_manager {
            if let Some(guest) = guest_manager.current_guest() {
                format!("Guest: {}", guest)
            } else {
                "Guest: None".to_string()
            }
        } else {
            "Guest: None".to_string()
        };
        writeln!(f, "{}", guest_line)?;

        writeln!(
            f,
            "Total Stats: S/P/C {}/{}/{}",
            self.total_team_smile, self.total_team_pure, self.total_team_cool
        )?;

        let slot_details: Vec<String> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                if slot.card_entry.is_some() {
                    Some(format!("\n[ Slot {} ]\n{}", i + 1, slot))
                } else {
                    None
                }
            })
            .collect();

        if slot_details.is_empty() {
            write!(f, "<Team is empty>")?;
        } else {
            write!(f, "{}", slot_details.join("\n"))?;
            write!(f, "\n--------------------------")?;
        }

        Ok(())
    }
}

fn _calculate_extra_skill_bonus(
    ls: &LeaderSkill,
    ts: &TeamSlot,
    additional_skill_map: &HashMap<String, HashSet<String>>,
) -> HashMap<String, u32> {
    let mut bonuses = HashMap::new();
    if let (Some(card), Some(attr), Some(target)) =
        (ts.card(), ls.extra_attribute(), ls.extra_target())
        && let Some(group) = additional_skill_map.get(target)
        && group.contains(&card.card_data.character)
    {
        let src = match attr.as_str() {
            "Smile" => ts.total_smile,
            "Pure" => ts.total_pure,
            "Cool" => ts.total_cool,
            _ => 0,
        };
        bonuses.insert(attr.clone(), (src as f32 * ls.extra_value()).ceil() as u32);
    }
    bonuses
}
