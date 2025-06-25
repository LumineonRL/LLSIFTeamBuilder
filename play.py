import math
import warnings
from typing import Optional, Dict, Set, List, Tuple, Any
from enum import IntEnum
from dataclasses import dataclass, field
import os
import json
import time
import logging

import numpy as np

from card import Card
from sis import SIS
from team import Team
from song import Song
from note import Note

class GameData:
    """
    Loads and holds all static game data from JSON files.
    This object is intended to be created once and shared across simulations.
    The 'too-few-public-methods' warning is disabled as this class is
    intentionally designed as a data container.
    """
    HEAL_MULTIPLIER = 480

    def __init__(self, data_path: str = 'data'):
        """
        Initializes GameData by loading all necessary files.
        Args:
            data_path: The path to the directory containing data files.
        """
        self.group_mapping = self._load_json_mapping(
            os.path.join(data_path, 'additional_leader_skill_map.json'),
            "additional leader skill map for PPN calculation"
        )
        self.sub_group_mapping = self._load_json_mapping(
            os.path.join(data_path, 'year_group_target.json'),
            "year group mapping for skills"
        )
        self.year_group_mapping = self._load_json_mapping(
            os.path.join(data_path, 'year_group_mapping.json'),
            "year group mapping for skills"
        )
        self.combo_bonus_tiers = self._load_combo_bonuses(
            os.path.join(data_path, 'combo_bonuses.json')
        )
        self.note_speed_map = self._load_note_speed_map(
            os.path.join(data_path, 'note_speed_map.json')
        )

    def _load_json_mapping(self, filepath: str, name: str) -> Dict[str, Set[str]]:
        """Generic helper to load a JSON file mapping groups to character sets."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            return {group.strip(): set(characters) for group, characters in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            warnings.warn(f"Could not load {name} from '{filepath}': {e}.")
            return {}

    def _load_combo_bonuses(self, filepath: str) -> List[Tuple[int, float]]:
        """Loads and sorts combo bonus tiers from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            return sorted([(int(k), v) for k, v in raw_mapping.items()], reverse=True)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load combo bonuses from '{filepath}': {e}.")
            return [(0, 1.0)]

    def _load_note_speed_map(self, filepath: str) -> Dict[int, float]:
        """Loads the note speed to on-screen duration mapping."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            return {int(k): v for k, v in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load note speed map from '{filepath}': {e}.")
            return {i: (1.6 - i * 0.1) if i >= 6 else (1.9 - i * 0.15) for i in range(1, 11)}

@dataclass(frozen=True)
class PlayConfig:
    """
    Holds the user-defined parameters for a simulation run.
    """
    accuracy: float = 0.9
    approach_rate: int = 9
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("Accuracy must be between 0.0 and 1.0.")
        if not 1 <= self.approach_rate <= 10:
            raise ValueError("Approach rate must be an integer between 1 and 10.")

class EventType(IntEnum):
    """
    Defines the types of events that can occur in the simulation.
    The numeric values define processing order for simultaneous events.
    Lower numbers are processed first.
    """
    SKILL_END = 1
    NOTE_SPAWN = 2
    TIME_SKILL = 3
    NOTE_START = 4
    NOTE_COMPLETION = 5
    SONG_END = 99 # Processed last

@dataclass(order=True)
class Event:
    """
    Represents a single, time-stamped event in the simulation.
    The `priority` field ensures that simultaneous events are processed in a
    deterministic and logical order (e.g., skill expirations before note scoring).
    """
    time: float
    priority: EventType
    payload: Any = field(default=None, compare=False)

class Trial:
    """
    Manages the state and execution of a single simulation trial.
    An instance of this class is created for each run within Play.simulate().
    """
    def __init__(self, play_instance: 'Play', random_state: np.random.Generator):
        # --- Static References ---
        self.play: 'Play' = play_instance
        self.team: 'Team' = play_instance.team
        self.song: 'Song' = play_instance.song
        self.config = play_instance.config
        self.game_data: 'GameData' = play_instance.game_data
        self.logger = play_instance.logger
        self.random_state = random_state

        # --- Dynamic State ---
        self.total_score: int = 0
        self.combo_count: int = 0
        self.perfect_hits: int = 0
        self.notes_hit: int = 0
        self.spawn_events_processed: int = 0

        self.current_slot_ppn: List[int] = list(play_instance.base_slot_ppn)

        self.active_pl_count: int = 0
        self.pl_uptime_start_time: Optional[float] = None
        self.uptime_intervals: List[Tuple[float, float]] = []

        self.total_trick_end_time: float = 0.0

        self.song_has_ended: bool = False

        # --- Event Queue & Song End Time ---
        self.event_queue, self.song_end_time = self._build_event_queue()

        # --- Trackers ---
        self.last_skill_info: Optional[Dict] = None
        self.score_skill_trackers = self._initialize_score_skill_trackers()
        self.year_group_skill_trackers = self._initialize_year_group_trackers()
        self.hold_note_start_results: Dict[int, str] = {}

    def run(self):
        """Executes the event loop for this trial until the queue is empty or the song ends."""
        while self.event_queue and not self.song_has_ended:
            event = self.event_queue.pop(0)
            self._dispatch(event)

    def _dispatch(self, event: Event):
        """Routes an event to its appropriate handler method."""
        handlers = {
            EventType.NOTE_SPAWN: self._handle_note_spawn,
            EventType.TIME_SKILL: self._handle_time_skill,
            EventType.SKILL_END: self._handle_skill_end,
            EventType.NOTE_START: self._handle_note_start,
            EventType.NOTE_COMPLETION: self._handle_note_completion,
            EventType.SONG_END: self._handle_song_end,
        }
        handler = handlers.get(event.priority)
        if handler:
            handler(event)

    # --- Initial State Setup ---

    def _build_event_queue(self) -> Tuple[List[Event], float]:
        """Creates and sorts the initial list of all events for the song."""
        events: List[Event] = []
        on_screen_duration = self.game_data.note_speed_map.get(self.config.approach_rate, 1.0)

        last_note_completion_time = 0.0
        if self.song.notes:
            last_note_completion_time = max(note.end_time for note in self.song.notes)

        for i, note in enumerate(self.song.notes):
            start_spawn_time = note.start_time - on_screen_duration
            events.append(Event(start_spawn_time, EventType.NOTE_SPAWN, payload={'note_idx': i, 'spawn_type': 'start'}))

            if note.start_time != note.end_time: # Hold note
                end_spawn_time = note.end_time - on_screen_duration
                events.append(Event(end_spawn_time, EventType.NOTE_SPAWN, payload={'note_idx': i, 'spawn_type': 'end'}))
                events.append(Event(note.start_time, EventType.NOTE_START, payload={'note_idx': i}))

            events.append(Event(note.end_time, EventType.NOTE_COMPLETION, payload={'note_idx': i}))

        song_end_time = last_note_completion_time + 0.001
        events.append(Event(song_end_time, EventType.SONG_END))

        for slot_idx, slot in enumerate(self.team.slots):
            if slot.card and slot.card.skill.activation == "Time":
                threshold = slot.card.skill_threshold or 0
                if threshold > 0:
                    for t in np.arange(threshold, self.song.length, threshold):
                        if t < song_end_time:
                            payload = {'slot_idx': slot_idx, 'card': slot.card}
                            events.append(Event(float(t), EventType.TIME_SKILL, payload=payload))

        events.sort()
        return events, song_end_time

    def _initialize_score_skill_trackers(self) -> Dict[int, int]:
        """Sets up the initial score thresholds for score-based skills."""
        return {
            idx: s.card.skill_threshold
            for idx, s in enumerate(self.team.slots)
            if s.card and s.card.skill.activation == "Score" and s.card.skill_threshold
        }

    def _initialize_year_group_trackers(self) -> Dict[int, Set[str]]:
        """Sets up trackers for 'Year Group' skills."""
        trackers = {}
        for idx, s in enumerate(self.team.slots):
            card = s.card
            if card and card.skill.activation == "Year Group" and card.skill.target:
                all_members = self.game_data.sub_group_mapping.get(card.skill.target, set())
                required_members = set(all_members) - {card.character}
                trackers[idx] = required_members
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "YEAR GROUP DBG: Initialized tracker for (%d) %s. Waiting for: %s",
                        idx + 1, card.display_name, required_members or "{None}"
                    )
        return trackers

    # --- Event Handlers ---
    def _handle_note_spawn(self, event: Event):
        if self.logger:
            note_idx = event.payload['note_idx']
            spawn_type = event.payload['spawn_type']
            self.logger.info(
                "EVENT @ %.3fs: Note #%d %s spawns.",
                event.time, note_idx + 1, spawn_type
            )
        self.spawn_events_processed += 1
        self._process_counter_skill("Rhythm Icons", self.spawn_events_processed, event.time)

    def _handle_time_skill(self, event: Event):
        card = event.payload['card']
        slot_idx = event.payload['slot_idx']
        if self.logger:
            self.logger.info(
                "EVENT @ %.3fs: Checking time skill for (%d) %s.",
                event.time, slot_idx + 1, card.display_name
            )
        if self.random_state.random() <= (card.skill_chance or 0.0):
            self._activate_skill(card, slot_idx, "Time", event.time)

    def _handle_skill_end(self, event: Event):
        if event.payload['type'] == 'pl_end':
            if self.logger:
                self.logger.info("EVENT @ %.3fs: A Perfect Lock effect ended.", event.time)
            self.active_pl_count -= 1
            if self.active_pl_count == 0:
                self._recalculate_ppn_with_tricks(active=False)
                if self.pl_uptime_start_time is not None:
                    interval_end = min(event.time, self.song_end_time)
                    interval = (self.pl_uptime_start_time, interval_end)
                    self.uptime_intervals.append(interval)
                    self.pl_uptime_start_time = None

    def _handle_note_start(self, event: Event):
        note_idx = event.payload['note_idx']
        if self.logger:
            self.logger.info("EVENT @ %.3fs: Processing hold note start for Note #%d.", event.time, note_idx + 1)

        if self.random_state.random() <= self.config.accuracy:
            self.hold_note_start_results[note_idx] = "Perfect"
            self.perfect_hits += 1
            self._process_counter_skill("Perfects", self.perfect_hits, event.time)
        else:
            self.hold_note_start_results[note_idx] = "Great"

    def _handle_note_completion(self, event: Event):
        note_idx = event.payload['note_idx']
        note = self.song.notes[note_idx]
        current_time = event.time

        is_pl = self.active_pl_count > 0
        is_trick = current_time <= self.total_trick_end_time
        original_hit_is_perfect = self.random_state.random() <= self.config.accuracy
        start_hit_is_perfect = self.hold_note_start_results.get(note_idx) != "Great"
        final_original_perfect = original_hit_is_perfect and start_hit_is_perfect

        if is_pl:
            hit_type = "Perfect"
            accuracy_multiplier = 1.08 if final_original_perfect else 1.0
        elif is_trick:
            hit_type = "Perfect"
            accuracy_multiplier = 1.0
        else:
            hit_type = "Perfect" if final_original_perfect else "Great"
            accuracy_multiplier = 1.0 if hit_type == "Perfect" else 0.88

        if hit_type == "Perfect":
            self.perfect_hits += 1
            self._process_counter_skill("Perfects", self.perfect_hits, current_time)

        hitting_slot_index = note.position - 1
        if 0 <= hitting_slot_index < len(self.team.slots):
            base_ppn = self.current_slot_ppn[hitting_slot_index]
            note_mult = self.play.get_note_multiplier(note)
            combo_mult = self.play.get_combo_multiplier(self.combo_count, self.game_data)

            note_score = math.floor(base_ppn * note_mult * combo_mult * accuracy_multiplier)
            self.total_score += note_score

            hitting_card = self.team.slots[hitting_slot_index].card
            if self.logger and hitting_card:
                self.logger.info(
                    "(%d) %s hit a %s on note #%d for %d points at %.3fs.",
                    hitting_slot_index + 1, hitting_card.display_name,
                    hit_type, note_idx + 1, note_score, current_time
                )

        self.notes_hit += 1
        self.combo_count += 1
        self._process_counter_skill("Combo", self.combo_count, current_time)
        if note.is_star:
            self._process_star_note_triggers(current_time)
        self._process_score_triggers(current_time)

    def _handle_song_end(self, event: Event):
        if self.logger:
            self.logger.info("EVENT @ %.3fs: Song has officially ended. No more events will be processed.", event.time)
        self.song_has_ended = True

    # --- Skill Activation Logic ---
    def _activate_skill(self, card: 'Card', slot_index: int, activation_type: str, current_time: float, copied_info: Optional[Dict] = None):
        """
        Handles the logic for a single skill activation.
        """
        # --- Encore Activation ---
        if activation_type == "Encore":
            if not copied_info:
                if self.logger:
                    self.logger.info("ENCORE: Encore activated with no skill to copy.")
                return

            copied_type = copied_info.get('type')
            score_gain = copied_info.get('score_gain', 0)
            duration = copied_info.get('duration', 0)

            if self.logger:
                log_details = ""
                if score_gain > 0:
                    log_details = f" for {score_gain} points"
                elif duration > 0:
                    log_details = f" for {duration:.2f} seconds"

                self.logger.info(
                    "SKILL: (%d) %s's Encore copies %s from (%d) %s%s.",
                    slot_index + 1, card.display_name, copied_type,
                    copied_info['slot_index'] + 1, copied_info['card'].display_name,
                    log_details
                )

            if copied_type == "Perfect Lock":
                if self.active_pl_count == 0:
                    self.pl_uptime_start_time = current_time
                    self._recalculate_ppn_with_tricks(active=True)
                self.active_pl_count += 1
                skill_end_time = min(current_time + duration, self.song_end_time)
                self.event_queue.append(Event(skill_end_time, EventType.SKILL_END, payload={'type': 'pl_end'}))
                self.event_queue.sort()

            if score_gain > 0:
                self.total_score += score_gain
                self._process_score_triggers(current_time)

            self._process_year_group_triggers(card.character, current_time)
            self.last_skill_info = {'type': 'Encore'}
            return

        # --- Standard Skill Activation ---
        score_gain = 0
        skill_type = card.skill.type
        skill_value = card.skill_value or 0
        duration = card.skill_duration or 0
        log_msg = ""

        if skill_type == "Scorer":
            charm_multiplier = 1.0
            for sis_entry in self.team.slots[slot_index].sis_entries:
                if sis_entry.sis.effect == "charm":
                    charm_multiplier *= sis_entry.sis.value
            score_gain = int(skill_value * charm_multiplier)
        elif skill_type == "Healer":
            slot = self.team.slots[slot_index]
            if any(s.sis.effect == "heal" for s in slot.sis_entries):
                score_gain = int(skill_value * self.game_data.HEAL_MULTIPLIER)
        elif skill_type == "Perfect Lock":
            if self.active_pl_count == 0:
                self.pl_uptime_start_time = current_time
                self._recalculate_ppn_with_tricks(active=True)
            self.active_pl_count += 1
            skill_end_time = min(current_time + duration, self.song_end_time)
            self.event_queue.append(Event(skill_end_time, EventType.SKILL_END, payload={'type': 'pl_end'}))
            self.event_queue.sort()
            log_msg = f"activated for {duration:.2f} seconds."
        elif skill_type == "Total Trick":
            new_end_time = min(current_time + duration, self.song_end_time)
            self.total_trick_end_time = max(self.total_trick_end_time, new_end_time)
            log_msg = f"activated for {duration:.2f} seconds."

        self.last_skill_info = {
            'card': card,
            'slot_index': slot_index,
            'type': skill_type,
            'score_gain': score_gain,
            'duration': duration
        }

        if self.logger:
            if score_gain > 0:
                log_msg = f"({activation_type}) activated for {score_gain} points"
            if log_msg:
                self.logger.info("SKILL: (%d) %s's skill %s", slot_index + 1, card.display_name, log_msg)

        if score_gain > 0:
            self.total_score += score_gain
            self._process_score_triggers(current_time)

        self._process_year_group_triggers(card.character, current_time)

    def _process_counter_skill(self, activation_type: str, counter: int, current_time: float):
        """
        Processes skills based on a counter (notes, combo, perfects).
        """
        for i, slot in enumerate(self.team.slots):
            card = slot.card
            if not card:
                continue

            if card.skill.activation == activation_type:
                threshold = card.skill_threshold
                if threshold and counter > 0 and counter % threshold == 0:
                    if self.random_state.random() <= (card.skill_chance or 0.0):

                        if card.skill.type == "Encore":
                            if self.last_skill_info and self.last_skill_info.get('type') != 'Encore':
                                self._activate_skill(card, i, "Encore", current_time, copied_info=self.last_skill_info)
                            elif self.logger:
                                self.logger.info(
                                    "SKILL: (%d) %s's Encore triggered but had nothing valid to copy.",
                                    i + 1, card.display_name
                                )
                        else:
                            self._activate_skill(card, i, activation_type, current_time)

    def _process_star_note_triggers(self, current_time: float):
        for i, slot in enumerate(self.team.slots):
            card = slot.card
            if card and card.skill.activation == "Star Notes":
                if self.random_state.random() <= (card.skill_chance or 0.0):
                    self._activate_skill(card, i, "Star Notes", current_time)

    def _process_score_triggers(self, current_time: float):
        activated_in_pass = True
        while activated_in_pass:
            activated_in_pass = False
            for idx, card in [(i, s.card) for i, s in enumerate(self.team.slots) if s.card]:
                if card.skill.activation == "Score":
                    threshold = card.skill_threshold
                    if not threshold:
                        continue
                    next_threshold = self.score_skill_trackers.get(idx, threshold)
                    if self.total_score >= next_threshold:
                        if self.random_state.random() <= (card.skill_chance or 0.0):
                            self._activate_skill(card, idx, "Score", current_time)
                            activated_in_pass = True
                        self.score_skill_trackers[idx] = next_threshold + threshold

    def _process_year_group_triggers(self, activating_character: str, current_time: float):
        for receiver_idx, required in list(self.year_group_skill_trackers.items()):
            if activating_character in required:
                receiver_card = self.team.slots[receiver_idx].card
                required.remove(activating_character)
                if self.logger and receiver_card and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "YEAR GROUP DBG: '%s' activated. Removing from tracker for (%d) %s. Remaining: %s",
                        activating_character, receiver_idx + 1, receiver_card.display_name, required or "{None}"
                    )

                if not required and receiver_card and receiver_card.skill.target:
                    if self.random_state.random() <= (receiver_card.skill_chance or 0.0):
                        self._activate_skill(receiver_card, receiver_idx, "Year Group", current_time)

                    target_group_name = receiver_card.skill.target
                    target_group = self.game_data.sub_group_mapping.get(target_group_name, set())
                    new_required = set(target_group) - {receiver_card.character}
                    self.year_group_skill_trackers[receiver_idx] = new_required
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            "YEAR GROUP DBG: Condition met for (%d) %s. Tracker reset. Waiting for: %s",
                            receiver_idx + 1, receiver_card.display_name, new_required or "{None}"
                        )

    def _recalculate_ppn_with_tricks(self, active: bool):
        if not active:
            self.current_slot_ppn = list(self.play.base_slot_ppn)
            return

        temp_stats = {
            "Smile": self.team.total_team_smile,
            "Pure": self.team.total_team_pure,
            "Cool": self.team.total_team_cool
        }
        for slot_idx, tricks in self.play.trick_slots.items():
            slot = self.team.slots[slot_idx]
            for trick_sis in tricks:
                base_stat = getattr(slot, f"total_{trick_sis.attribute.lower()}")
                bonus = math.ceil(base_stat * trick_sis.value)
                temp_stats[trick_sis.attribute] += bonus
        new_team_total = temp_stats.get(self.song.attribute, 0)
        self.current_slot_ppn = self.play.calculate_ppn_for_all_slots(new_team_total)

    def get_total_pl_uptime(self) -> float:
        """Calculates the percentage of a song that perfect lock skills were active"""
        if not self.uptime_intervals:
            return 0.0
        intervals = sorted(self.uptime_intervals)
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        return sum(end - start for start, end in merged)

# --- Main Public Class ---
# The `Play` class is now a high-level orchestrator. It sets up the simulation
# but delegates the complex execution logic to the `Trial` class.

class Play:
    """
    Represents an instance of a song being played by a specific team.
    This class orchestrates the simulation but delegates the heavy lifting
    of a single run to the `Trial` class.
    """
    def __init__(
        self,
        team: Team,
        song: Song,
        config: PlayConfig,
        game_data: GameData
    ):
        """
        Initializes a new play instance.

        Args:
            team: The Team object playing the song.
            song: The Song object being played.
            config: A PlayConfig object with simulation parameters.
            game_data: A GameData object with static game data.
        """
        self.team = team
        self.song = song
        self.config = config
        self.game_data = game_data
        self.random_state = np.random.default_rng(config.seed)
        self.logger: Optional[logging.Logger] = None

        # Pre-calculate values that are constant across all trials
        self.trick_slots: Dict[int, List[SIS]] = {
            i: [s.sis for s in slot.sis_entries if s.sis.effect == "trick"]
            for i, slot in enumerate(self.team.slots) if slot.card
        }
        team_total_stat = getattr(self.team, f"total_team_{self.song.attribute.lower()}", 0)
        self.base_slot_ppn: List[int] = self.calculate_ppn_for_all_slots(team_total_stat)

    def simulate(self, n: int = 1, log_level: int = logging.DEBUG) -> None:
        """
        Runs the simulation for n trials.

        Args:
            n: The number of trials to run.
            log_level: The logging level to use (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger = self._setup_logger(log_level)

        trial_scores, trial_uptimes = [], []

        for i in range(n):
            print(f"--- Starting Simulation Trial {i + 1}/{n} ---")
            if self.logger:
                self.logger.info("--- Simulation Trial %d/%d ---", i + 1, n)

            trial = Trial(self, self.random_state)
            trial.run()

            trial_scores.append(trial.total_score)
            total_uptime = trial.get_total_pl_uptime()
            trial_uptimes.append(total_uptime)
            self._log_trial_summary(trial, total_uptime)

            print(f"--- Trial {i + 1} Finished. Final Score: {trial.total_score} ---")

        if n > 1:
            self._log_overall_summary(trial_scores, trial_uptimes)

    # --- PPN and Multiplier Calculation Helpers ---
    def _check_group_bonus(self, card: Card) -> float:
        """Checks if a card's group matches the song's group for a 0.1 bonus."""
        song_group = self.song.group
        valid_members = self.game_data.group_mapping.get(song_group, set())
        return 0.1 if card.character in valid_members else 0.0

    def _check_attribute_bonus(self, card: Card) -> float:
        """Checks if a card's attribute matches the song's for a 0.1 bonus."""
        return 0.1 if self.song.attribute == card.attribute else 0.0

    def calculate_ppn_for_all_slots(self, team_total_stat: int) -> List[int]:
        """Calculates the PPN for each team slot given a total team stat."""
        if team_total_stat == 0:
            warnings.warn("Team total stat for song attribute is 0. All PPN values will be 0.")
            return [0] * self.team.NUM_SLOTS

        ppn_values = []
        for slot in self.team.slots:
            if not slot.card:
                ppn_values.append(0)
                continue
            group_bonus = self._check_group_bonus(slot.card)
            attribute_bonus = self._check_attribute_bonus(slot.card)
            slot_ppn = math.floor(team_total_stat * 0.0125 * (1 + group_bonus + attribute_bonus))
            ppn_values.append(slot_ppn)
        return ppn_values

    @staticmethod
    def get_note_multiplier(note: Note) -> float:
        """Determines the score multiplier for a given note type."""
        is_hold = note.start_time != note.end_time
        is_swing = note.is_swing
        if is_hold and is_swing:
            return 0.625
        if is_hold:
            return 1.25
        if is_swing:
            return 0.5
        return 1.0

    @staticmethod
    def get_combo_multiplier(combo_count: int, game_data: GameData) -> float:
        """Finds the combo multiplier for the current combo count."""
        for threshold, multiplier in game_data.combo_bonus_tiers:
            if combo_count + 1 >= threshold:
                return multiplier
        return 1.0

    # --- Logging and Output ---
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Configures and returns a logger for the simulation."""
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        sanitized_title = "".join(c for c in self.song.title if c.isalnum() or c in (' ', '_')).rstrip()
        timestamp = int(time.time())
        log_filename = f"{sanitized_title.replace(' ', '_')}_{self.song.difficulty}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        logger = logging.getLogger(log_filepath)
        logger.setLevel(log_level)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        handler = logging.FileHandler(log_filepath, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger

    def _log_trial_summary(self, trial: Trial, total_uptime: float):
        if not self.logger:
            return

        self.logger.info("--- Trial Final Score: %d ---", trial.total_score)
        hold_starts = len(trial.hold_note_start_results)
        total_judgements = trial.notes_hit + hold_starts
        ratio_percent = (trial.perfect_hits / total_judgements * 100) if total_judgements > 0 else 0
        self.logger.info("Perfect Ratio: %d / %d (%.2f%%)", trial.perfect_hits, total_judgements, ratio_percent)
        uptime_percent = (total_uptime / self.song.length * 100) if self.song.length > 0 else 0
        self.logger.info("Perfect Lock Uptime: %.2fs (%.2f%%)", total_uptime, uptime_percent)

    def _log_overall_summary(self, scores: List[int], uptimes: List[float]):
        if not self.logger:
            return

        self.logger.info("\n--- Overall Simulation Summary ---")
        self.logger.info("Average Score: %.0f", np.mean(scores))
        self.logger.info("Max Score: %d", np.max(scores))
        self.logger.info("Min Score: %d", np.min(scores))
        self.logger.info("Standard Deviation: %.2f", np.std(scores))
        avg_uptime_percent = np.mean([t / self.song.length * 100 for t in uptimes]) if self.song.length > 0 else 0
        self.logger.info("Average Perfect Lock Uptime: %.2fs (%.2f%%)", np.mean(uptimes), avg_uptime_percent)

    def __repr__(self) -> str:
        """Provides a summary of the Play configuration."""
        return (
            f"<Play Configuration>\n"
            f"  - Song: '{self.song.title}' ({self.song.difficulty})\n"
            f"  - Team Stats (S/P/C): {self.team.total_team_smile}/{self.team.total_team_pure}/{self.team.total_team_cool}\n"
            f"  - Accuracy: {self.config.accuracy:.2%}\n"
            f"  - Approach Rate: {self.config.approach_rate}\n"
            f"  - Base PPN: {self.base_slot_ppn}"
        )
