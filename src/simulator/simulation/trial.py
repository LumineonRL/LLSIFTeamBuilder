"""
This module manages the state and execution of a single simulation trial by
orchestrating specialized handlers.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from src.simulator.simulation.events import Event, EventType
from src.simulator.simulation.event_processor import EventProcessor
from src.simulator.simulation.trial_state import TrialState

if TYPE_CHECKING:
    from src.simulator.simulation.play import Play


class Trial:
    """
    Manages the setup and execution of a single simulation trial.

    This class initializes the trial's state and its various logic handlers,
    builds the initial event queue, and runs the main event loop. It delegates
    all event-specific logic to the EventProcessor.
    """

    def __init__(self, play_instance: "Play", random_state: np.random.Generator):
        """Initializes a single simulation trial."""
        # --- Static References & Context ---
        self.play = play_instance
        self.team = play_instance.team
        self.song = play_instance.song
        self.config = play_instance.config
        self.game_data = play_instance.game_data
        self.logger = play_instance.logger

        # --- Dynamic State ---
        self.state = TrialState(random_state=random_state)
        self.state.current_slot_ppn = list(play_instance.base_slot_ppn)
        self._initialize_trackers()

        # --- Event Queue & Processor ---
        self.event_queue, self.song_end_time = self._build_event_queue()
        self.state.song_end_time = self.song_end_time

        processor_context = {
            "game_data": self.game_data,
            "logger": self.logger,
            "random_state": random_state,
            "play": play_instance,
        }
        self.processor = EventProcessor(processor_context)

    def run(self):
        """
        Executes the event loop for this trial until the queue is empty or
        the song has ended.
        """
        while self.event_queue and not self.state.song_has_ended:
            event = heapq.heappop(self.event_queue)
            self.processor.dispatch(event, self.state, self.play, self.event_queue)

    def _initialize_trackers(self):
        """Sets up the initial state for various skill trackers."""
        self.state.score_skill_trackers = {
            idx: s.card.skill_threshold
            for idx, s in enumerate(self.team.slots)
            if s.card and s.card.skill.activation == "Score" and s.card.skill_threshold
        }

        # Initialize trackers for Year Group skills
        for idx, s in enumerate(self.team.slots):
            card = s.card
            if not (
                card and card.skill.activation == "Year Group" and card.skill.target
            ):
                continue

            all_members = self.game_data.sub_group_mapping.get(card.skill.target, set())
            required_members = set(all_members) - {card.character}
            self.state.year_group_skill_trackers[idx] = required_members
            if self.logger:
                self.logger.debug(
                    "YEAR GROUP DBG: Initialized tracker for (%d) %s. "
                    "Waiting for: %s",
                    idx + 1,
                    card.display_name,
                    required_members or "{None}",
                )

    def _build_event_queue(self) -> Tuple[List[Event], float]:
        """Creates the initial list of all events for the song."""
        events: List[Event] = []
        on_screen_duration = self.game_data.note_speed_map.get(
            self.config.approach_rate, 1.0
        )

        last_note_completion_time = 0.0
        if self.song.notes:
            last_note_completion_time = max(note.end_time for note in self.song.notes)

        for i, note in enumerate(self.song.notes):
            # Note spawn events
            heapq.heappush(
                events,
                Event(
                    time=note.start_time - on_screen_duration,
                    priority=EventType.NOTE_SPAWN,
                    payload={"note_idx": i, "spawn_type": "start"},
                ),
            )

            if note.start_time != note.end_time:  # Hold note
                heapq.heappush(
                    events,
                    Event(
                        time=note.end_time - on_screen_duration,
                        priority=EventType.NOTE_SPAWN,
                        payload={"note_idx": i, "spawn_type": "end"},
                    ),
                )
                heapq.heappush(
                    events,
                    Event(
                        time=note.start_time,
                        priority=EventType.NOTE_START,
                        payload={"note_idx": i},
                    ),
                )

            # Note completion event
            heapq.heappush(
                events,
                Event(
                    time=note.end_time,
                    priority=EventType.NOTE_COMPLETION,
                    payload={"note_idx": i},
                ),
            )

        # Song end event
        song_end_time = last_note_completion_time + 0.001
        heapq.heappush(events, Event(song_end_time, EventType.SONG_END))

        # Time-based skill events
        for slot_idx, slot in enumerate(self.team.slots):
            if slot.card and slot.card.skill.activation == "Time":
                threshold = slot.card.skill_threshold or 0
                if threshold > 0:
                    for t in np.arange(threshold, self.song.length, threshold):
                        payload = {"card": slot.card, "slot_idx": slot_idx}
                        heapq.heappush(
                            events,
                            Event(float(t), EventType.TIME_SKILL, payload=payload),
                        )

        return events, song_end_time

    def get_total_pl_uptime(self) -> float:
        """Calculates the total merged uptime for Perfect Lock effects."""
        intervals = self.state.uptime_intervals
        if not intervals:
            return 0.0

        # Sort and merge overlapping intervals
        intervals.sort()
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return sum(end - start for start, end in merged)

    @property
    def total_score(self) -> int:
        """Convenience property to access the final score from the state."""
        return self.state.total_score

    @property
    def perfect_hits(self) -> int:
        """Convenience property to access perfect hits from the state."""
        return self.state.perfect_hits

    @property
    def notes_hit(self) -> int:
        """Convenience property to access total notes hit from the state."""
        return self.state.notes_hit

    @property
    def hold_note_start_results(self) -> Dict[int, str]:
        """Convenience property to access hold start judgements from state."""
        return self.state.hold_note_start_results
