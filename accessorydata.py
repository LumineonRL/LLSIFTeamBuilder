from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class AccessoryData:
    """Represents the static, immutable data for an accessory from JSON."""
    accessory_id: int
    name: str
    character: str
    card_id: Optional[str] = None
    stats: List[List[int]] = field(default_factory=list)
    skill: Dict[str, Any] = field(default_factory=dict)
