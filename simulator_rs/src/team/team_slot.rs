use crate::accessory::{Accessory, PlayerAccessory};
use crate::card::{Card, PlayerCard};
use crate::sis::{PlayerSIS, SIS};
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Debug, Default)]
pub struct TeamSlot {
    pub card_entry: Option<PlayerCard>,
    pub accessory_entry: Option<PlayerAccessory>,
    pub sis_entries: Vec<PlayerSIS>,
    pub total_smile: u32,
    pub total_pure: u32,
    pub total_cool: u32,
}

impl TeamSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn card(&self) -> Option<&Card> {
        self.card_entry.as_ref().map(|pc| &pc.card)
    }

    pub fn accessory(&self) -> Option<&Accessory> {
        self.accessory_entry.as_ref().map(|pa| &pa.accessory)
    }

    pub fn sis_list(&self) -> Vec<&SIS> {
        self.sis_entries.iter().map(|ps| &*ps.sis).collect()
    }

    pub fn total_sis_slots_used(&self) -> u32 {
        self.sis_entries.iter().map(|ps| ps.sis.slots).sum()
    }

    pub fn card_sis_capacity(&self) -> u8 {
        self.card().map_or(0, |c| c.current_sis_slots())
    }

    pub fn available_sis_slots(&self) -> u8 {
        self.card_sis_capacity()
            .saturating_sub(self.total_sis_slots_used() as u8)
    }

    pub fn equip_card(&mut self, card_entry: PlayerCard) {
        self.clear();
        self.card_entry = Some(card_entry);
    }

    pub fn equip_accessory(&mut self, accessory_entry: PlayerAccessory) -> Result<(), &str> {
        if self.card_entry.is_none() {
            return Err("Cannot equip an accessory: No card is in this slot.");
        }
        self.accessory_entry = Some(accessory_entry);
        Ok(())
    }

    pub fn equip_sis(&mut self, sis_entry: PlayerSIS) -> Result<(), &str> {
        if self.card_entry.is_none() {
            return Err("Cannot equip SIS: No card is in this slot.");
        }
        self.sis_entries.push(sis_entry);
        Ok(())
    }

    pub fn unequip_sis(&mut self, manager_internal_id: u32) -> bool {
        let initial_count = self.sis_entries.len();
        self.sis_entries
            .retain(|ps| ps.manager_internal_id != manager_internal_id);
        self.sis_entries.len() < initial_count
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

impl Display for TeamSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let card_entry = match &self.card_entry {
            Some(ce) => ce,
            None => return write!(f, "  <Empty Slot>"),
        };

        write!(
            f,
            "  Card: {} (Deck ID: {})",
            card_entry.card.card_data.display_name, card_entry.manager_internal_id
        )?;
        write!(
            f,
            "\n  Stats: S/P/C {}/{}/{}",
            self.total_smile, self.total_pure, self.total_cool
        )?;

        if let Some(acc_entry) = &self.accessory_entry {
            write!(
                f,
                "\n  Accessory: {} (Manager ID: {})",
                acc_entry.accessory.data.name, acc_entry.manager_internal_id
            )?;
        } else {
            write!(f, "\n  Accessory: None")?;
        }

        if !self.sis_entries.is_empty() {
            write!(
                f,
                "\n  SIS ({}/{} slots used):",
                self.total_sis_slots_used(),
                self.card_sis_capacity()
            )?;
            for sis_entry in &self.sis_entries {
                write!(
                    f,
                    "\n    - {} ({} slots)",
                    sis_entry.sis.name, sis_entry.sis.slots
                )?;
            }
        } else {
            write!(f, "\n  SIS: None")?;
        }

        Ok(())
    }
}
