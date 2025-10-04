"""
This module defines the abstract interface for an event publisher.

It decouples event-generating components from the concrete event queue
implementation.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulator.simulation.events import Event


class EventPublisher(metaclass=ABCMeta):
    """
    An abstract interface for publishing events.

    Classes that manage an event queue can implement this interface to allow
    other components to add events without needing direct access to the queue.
    """

    @abstractmethod
    def publish(self, event: Event):
        """Adds an event to the event stream."""
        raise NotImplementedError
