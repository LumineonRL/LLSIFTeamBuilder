from typing import Optional

from sisdata import SISData

class SIS:
    """
    Represents an instance of a School Idol Skill.
    
    Since SIS objects do not have a mutable state like level, this class
    acts as a clean interface to the underlying SIS data.
    """
    def __init__(self, sis_data: SISData):
        self._data = sis_data

        self.id: int = self._data.id
        self.name: str = self._data.name
        self.effect: str = self._data.effect
        self.slots: int = self._data.slots
        self.attribute: str = self._data.attribute
        self.group: Optional[str] = self._data.group
        self.equip_restriction: Optional[str] = self._data.equip_restriction
        self.target: Optional[str] = self._data.target
        self.value: float = self._data.value

    def __repr__(self) -> str:
        """Provides a detailed representation of the SIS."""
        return (
            f"<SIS id={self.id} name='{self.name}'>\n"
            f"  - Effect: {self.effect} ({self.value})\n"
            f"  - Slots: {self.slots}, Attribute: {self.attribute}"
        )
