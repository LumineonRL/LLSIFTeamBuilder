from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class CardData:
    """Represents the static, immutable data for a card, loaded from JSON."""
    card_id: int
    display_name: str
    rarity: str
    attribute: str
    character: str
    is_promo: bool
    is_preidolized_non_promo: bool
    stats: Dict[str, Any]
    skill: Dict[str, Any]
    leader_skill: Dict[str, Any]
