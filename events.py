"""
This module defines the event system for the simulation.

It includes the types of events that can occur and the structure for
representing a single event in the simulation queue.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EventType(IntEnum):
    """
    Defines the types of events that can occur in the simulation.

    The numeric values define the processing priority for simultaneous events.
    Lower numbers are processed first, ensuring a deterministic and logical
    order (e.g., skill expirations are handled before new notes are scored).
    """
    LOCK_END = 1
    SYNC_END = 2
    SKILL_RATE_UP_END = 3
    APPEAL_BOOST_END = 4
    PERFECT_SCORE_UP_END = 5
    COMBO_BONUS_UP_END = 6
    SPARK_END = 7
    NOTE_SPAWN = 8
    TIME_SKILL = 9
    NOTE_START = 10
    NOTE_COMPLETION = 11
    SONG_END = 99


@dataclass(order=True)
class Event:
    """
    Represents a single, time-stamped event in the simulation.

    The `time` and `priority` fields are used for sorting, ensuring the
    event queue is always processed in the correct chronological and
    prioritized order.

    Attributes:
        time (float): The simulation time at which the event occurs.
        priority (EventType): The type of the event, which also serves as
                              its processing priority.
        payload (Any): Optional data associated with the event, such as a
                       note index or skill information. This field is not
                       used in sorting comparisons.
    """
    time: float
    priority: EventType
    payload: Any = field(default=None, compare=False)
