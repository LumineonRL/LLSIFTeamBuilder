use crate::sis::data::SisData;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Sis {
    pub data: Arc<SisData>,
}

impl Sis {
    pub fn new(data: Arc<SisData>) -> Self {
        Self { data }
    }

    // Convenience accessors that proxy to the underlying data
    pub fn id(&self) -> u16 {
        self.data.id
    }

    pub fn name(&self) -> &str {
        &self.data.name
    }

    pub fn effect(&self) -> &str {
        &self.data.effect
    }

    pub fn slots(&self) -> u8 {
        self.data.slots
    }

    pub fn attribute(&self) -> &str {
        &self.data.attribute
    }

    pub fn group(&self) -> Option<&str> {
        self.data.group.as_deref()
    }

    pub fn equip_restriction(&self) -> Option<&str> {
        self.data.equip_restriction.as_deref()
    }

    pub fn target(&self) -> Option<&str> {
        self.data.target.as_deref()
    }

    pub fn value(&self) -> f32 {
        self.data.value
    }
}

impl PartialEq for Sis {
    fn eq(&self, other: &Self) -> bool {
        self.data.id == other.data.id
    }
}

impl Display for Sis {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "<SIS id={} name='{}'>", self.id(), self.name())?;
        writeln!(f, "  - Effect: {} ({})", self.effect(), self.value())?;
        writeln!(
            f,
            "  - Slots: {}, Attribute: {}",
            self.slots(),
            self.attribute()
        )
    }
}
