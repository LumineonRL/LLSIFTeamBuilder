"""
This module defines the configuration for a simulation run.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlayConfig:
    """
    Holds the user-defined parameters for a simulation run.

    Using a dataclass simplifies passing configuration around and ensures
    immutability of simulation parameters during a run.

    Attributes:
        accuracy (float): The base probability (0.0 to 1.0) of hitting a
                          note with 'Perfect' timing. Defaults to 0.9.
        approach_rate (int): The note speed setting (1-10), affecting how
                             long notes are visible on screen. Defaults to 9.
        seed (Optional[int]): An optional seed for the random number
                              generator to ensure reproducibility.
                              Defaults to None.
        enable_logging (bool): Whether to write detailed simulation logs to file.
                               Defaults to False.
        log_level (int): The logging level to use when logging is enabled.
                         Defaults to logging.INFO.
    """

    accuracy: float = 0.9
    approach_rate: int = 9
    seed: Optional[int] = None
    enable_logging: bool = False
    log_level: int = 20

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("Accuracy must be between 0.0 and 1.0.")
        if not 1 <= self.approach_rate <= 10:
            raise ValueError("Approach rate must be an integer between 1 and 10.")
