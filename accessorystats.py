from dataclasses import dataclass

@dataclass(frozen=True)
class AccessoryStats:
    """Holds the core stats for a card in a specific idolization state."""
    smile: int = 0
    pure: int = 0
    cool: int = 0