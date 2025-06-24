import math
import warnings
from typing import Optional, Dict, Set, List, Tuple, Any
import os
import json
import time
import logging

import numpy as np

from team import Team
from song import Song
from card import Card
from teamslot import TeamSlot
from note import Note


class Play:
    """
    Represents an instance of a song being played by a specific team,
    handling the simulation of the gameplay and score calculation.
    """
    INCREMENT_RATE = 0.01

    def _load_json_mapping(self, filepath: str, warning_message: str) -> Dict[str, Set[str]]:
        """Generic helper to load a JSON file mapping groups to character sets."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            # Strip whitespace from keys to prevent matching errors
            return {group.strip(): set(characters) for group, characters in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            warnings.warn(f"{warning_message} from '{filepath}': {e}.")
            return {}

    def _load_combo_bonuses(self, filepath: str) -> List[Tuple[int, float]]:
        """Loads and processes the combo bonus mapping from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            # Convert string keys to int and sort by combo count descending
            return sorted([(int(k), v) for k, v in raw_mapping.items()], reverse=True)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load or parse combo bonuses from '{filepath}': {e}.")
            return [(0, 1.0)] # Default to no bonus if file is missing/invalid

    def _load_note_speed_map(self, filepath: str) -> Dict[int, float]:
        """Loads the note speed to on-screen duration mapping."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
            return {int(k): v for k, v in raw_mapping.items()}
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            warnings.warn(f"Could not load note speed map from '{filepath}': {e}.")
            # Provide a default map if loading fails, based on the formula
            return {i: (1.6 - i * 0.1) if i >= 6 else (1.9 - i * 0.15) for i in range(1, 11)}

    def __init__(
        self,
        team: Team,
        song: Song,
        accuracy: float = 0.9,
        approach_rate: int = 9,
        seed: Optional[int] = None
    ):
        """
        Initializes a new play instance.

        Args:
            team: The Team object playing the song.
            song: The Song object being played.
            accuracy: The percentage of perfects, from 0.00 to 1.00.
            approach_rate: The note approach rate, from 1 to 10.
            seed: An optional seed for reproducible randomness.
        """
        # --- Parameter Validation ---
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("Accuracy must be between 0.0 and 1.0.")
        if not 1 <= approach_rate <= 10:
            raise ValueError("Approach rate must be an integer between 1 and 10.")

        # --- Instance Attributes ---
        self.team = team
        self.song = song
        self.accuracy = accuracy
        self.approach_rate = approach_rate
        self.random_state = np.random.default_rng(seed)
        self.logger: Optional[logging.Logger] = None
        self.total_score: int = 0

        # --- Load Mappings ---
        self._group_mapping = self._load_json_mapping(
            os.path.join('data', 'additional_leader_skill_map.json'),
            "Could not load additional leader skill map for PPN calculation"
        )
        self._combo_bonus_tiers = self._load_combo_bonuses(
            os.path.join('data', 'combo_bonuses.json')
        )
        self._note_speed_map = self._load_note_speed_map(
            os.path.join('data', 'note_speed_map.json')
        )

        # --- Initial Calculations ---
        self.slot_ppn_values: List[int] = self._calculate_all_slot_ppns()
        self.note_spawn_events: List[Tuple[float, str, int]] = self._calculate_note_spawn_events()
        self._setup_logger()

    def _setup_logger(self):
        """Configures a logger to save simulation results to a unique file."""
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)

        # Sanitize song title and create unique filename
        sanitized_title = "".join(c for c in self.song.title if c.isalnum() or c in (' ', '_')).rstrip()
        timestamp = int(time.time())
        log_filename = f"{sanitized_title.replace(' ', '_')}_{self.song.difficulty}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        # Configure logger
        self.logger = logging.getLogger(log_filepath)
        self.logger.setLevel(logging.INFO)
        # Prevent logs from propagating to parent loggers (e.g., root logger)
        self.logger.propagate = False
        
        # Clear existing handlers to prevent duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _calculate_note_spawn_events(self) -> List[Tuple[float, str, int]]:
        """
        Calculates all note spawn times based on the approach rate.
        A spawn event is when a note first appears on screen.
        Returns a list of tuples: (spawn_time, event_type, note_index)
        """
        on_screen_duration = self._note_speed_map.get(self.approach_rate, 1.0)
        spawn_events: List[Tuple[float, str, int]] = []

        for i, note in enumerate(self.song.notes):
            # Calculate spawn time for the start of the note
            start_spawn_time = note.start_time - on_screen_duration
            spawn_events.append((start_spawn_time, 'start_spawn', i))
            
            # If it's a hold note, also calculate spawn time for the end
            if note.start_time != note.end_time:
                end_spawn_time = note.end_time - on_screen_duration
                spawn_events.append((end_spawn_time, 'end_spawn', i))
        
        # Sort all events chronologically
        return sorted(spawn_events)

    def _check_group(self, card: Card) -> float:
        """
        Checks if a card's character belongs to the song's group.
        Returns 0.1 if they match, 0.0 otherwise.
        """
        song_group = self.song.group
        valid_members = self._group_mapping.get(song_group, set())
        if card.character in valid_members:
            return 0.1
        return 0.0

    def _check_attribute(self, card: Card) -> float:
        """
        Checks if a card's attribute matches the song's attribute.
        Returns 0.1 if they match, 0.0 otherwise.
        """
        if self.song.attribute == card.attribute:
            return 0.1
        return 0.0
        
    def _calculate_all_slot_ppns(self) -> List[int]:
        """
        Calculates the PPN for each team slot and returns them as a list.
        The PPN for each slot is based on the team's total stat for the song's attribute,
        plus bonuses for the card in that specific slot.
        """
        ppn_values = []
        song_attribute_lower = self.song.attribute.lower()

        # Get the single team-wide stat to be used for all slots
        team_total_stat = 0
        if song_attribute_lower == "smile":
            team_total_stat = self.team.total_team_smile
        elif song_attribute_lower == "pure":
            team_total_stat = self.team.total_team_pure
        elif song_attribute_lower == "cool":
            team_total_stat = self.team.total_team_cool
        
        if team_total_stat == 0:
            warnings.warn("Team total stat for song attribute is 0. All PPN values will be 0.")
            return [0] * self.team.NUM_SLOTS

        for slot in self.team.slots:
            if not slot.card:
                ppn_values.append(0)
                continue

            # Check for slot-specific bonuses
            group_bonus = self._check_group(slot.card)
            attribute_bonus = self._check_attribute(slot.card)
            
            # Calculate PPN for this slot using the team's total stat
            slot_ppn = math.floor(team_total_stat * 0.0125 * (1 + group_bonus + attribute_bonus))
            ppn_values.append(slot_ppn)
            
        return ppn_values

    def _get_note_multiplier(self, note: Note) -> float:
        """Determines the score multiplier for a given note type."""
        is_hold = note.start_time != note.end_time
        is_swing = note.is_swing

        if is_hold and is_swing:
            return 0.625  # Hold-swing note
        elif is_hold:
            return 1.25   # Hold note
        elif is_swing:
            return 0.5    # Swing note
        else:
            return 1.0    # Regular tap note

    def _get_combo_multiplier(self, combo_count: int) -> float:
        """Finds the combo multiplier for the current combo count."""
        for threshold, multiplier in self._combo_bonus_tiers:
            # Since the list is sorted descending, the first match is the correct one.
            if combo_count + 1 >= threshold:
                return multiplier
        return 1.0 # Default case

    def _handle_note_completion(self, note: Note, original_note_index: int, combo_count: int, current_time: float) -> bool:
        """Processes a single note event when its end_time is reached."""
        # Determine hit type based on a new random number
        if self.random_state.random() <= self.accuracy:
            hit_type = "Perfect"
            accuracy_multiplier = 1.0
        else:
            hit_type = "Great"
            accuracy_multiplier = 0.88

        hitting_slot_index = note.position - 1
        if not (0 <= hitting_slot_index < len(self.team.slots)):
            return hit_type == "Perfect"

        hitting_slot = self.team.slots[hitting_slot_index]
        if hitting_slot.card and self.logger:
            base_ppn = self.slot_ppn_values[hitting_slot_index]
            note_type_multiplier = self._get_note_multiplier(note)
            combo_multiplier = self._get_combo_multiplier(combo_count)
            note_score = math.floor(base_ppn * note_type_multiplier * combo_multiplier * accuracy_multiplier)
            
            self.total_score += note_score
            self.logger.info(
                "(%d) %s hit a %s on note #%d for %d points at %.3fs.",
                hitting_slot_index + 1, hitting_slot.card.display_name,
                hit_type, original_note_index + 1, note_score, current_time
            )
        
        return hit_type == "Perfect"

    def _handle_note_spawn(self, spawn_events_processed: int):
        """Handles skill activations triggered by a note spawning."""
        for i, slot in enumerate(self.team.slots):
            card = slot.card
            if not (
                card
                and card.skill.type == "Scorer"
                and card.skill.activation == "Rhythm Icons"
            ):
                continue

            threshold = card.skill_threshold
            # Ensure skill can trigger (has threshold) and spawn count is a multiple of it
            if threshold and spawn_events_processed > 0 and spawn_events_processed % threshold == 0:
                roll = self.random_state.random()
                chance = card.skill_chance or 0.0
                if roll <= chance:
                    score_gain = card.skill_value or 0
                    self.total_score += int(score_gain)
                    if self.logger:
                        self.logger.info(
                            "(%d) %s's skill activated for %d points",
                            i + 1, card.display_name, int(score_gain)
                        )

    def simulate(self, n: int = 1) -> None:
        """
        Runs the simulation of the song play for 'n' trials.
        """
        notes_completion_events = sorted(enumerate(self.song.notes), key=lambda item: item[1].end_time)
        spawn_events = self.note_spawn_events
        
        trial_scores = []

        for i in range(n):
            print(f"--- Starting Simulation Trial {i + 1}/{n} ---")
            
            self.total_score = 0
            combo_count, perfect_hits, notes_hit, spawn_events_processed = 0, 0, 0, 0
            
            if self.logger:
                self.logger.info("--- Simulation Trial %d/%d ---", i + 1, n)

            notes_to_process = list(notes_completion_events)
            spawns_to_process = list(spawn_events)
            
            current_time = 0.0
            
            while spawns_to_process or notes_to_process:
                # Determine the time of the next event
                next_spawn_time = spawns_to_process[0][0] if spawns_to_process else float('inf')
                next_completion_time = notes_to_process[0][1].end_time if notes_to_process else float('inf')
                
                # If there are no events left, break
                if next_spawn_time == float('inf') and next_completion_time == float('inf'):
                    break

                # Advance time to the next event
                current_time = min(next_spawn_time, next_completion_time)

                # Process all spawn events at the current time
                while spawns_to_process and spawns_to_process[0][0] <= current_time:
                    _, _, note_idx = spawns_to_process.pop(0)
                    spawn_events_processed += 1
                    self._handle_note_spawn(spawn_events_processed)

                # Process all note completions at the current time
                while notes_to_process and notes_to_process[0][1].end_time <= current_time:
                    original_index, note_to_complete = notes_to_process.pop(0)
                    if self._handle_note_completion(note_to_complete, original_index, combo_count, current_time):
                        perfect_hits += 1
                    
                    combo_count += 1
                    notes_hit += 1

            trial_scores.append(self.total_score)
            
            if self.logger:
                self.logger.info("--- Trial %d Final Score: %d ---", i + 1, self.total_score)
                ratio_percent = (perfect_hits / notes_hit * 100) if notes_hit > 0 else 0
                self.logger.info("Perfect Ratio: %d / %d (%.2f%%)", perfect_hits, notes_hit, ratio_percent)

            print(f"--- Trial {i + 1} Finished. Final Score: {self.total_score} ---")
            
        # Log summary statistics if more than one trial was run
        if n > 1 and self.logger:
            self.logger.info("\n--- Overall Simulation Summary ---")
            self.logger.info("Average Score: %.0f", np.mean(trial_scores))
            self.logger.info("Max Score: %d", np.max(trial_scores))
            self.logger.info("Min Score: %d", np.min(trial_scores))
            self.logger.info("Standard Deviation: %.2f", np.std(trial_scores))

    def __repr__(self) -> str:
        """Provides a summary of the Play configuration."""
        header = "<Play Configuration>"
        details = (
            f"  - Song: '{self.song.title}' ({self.song.difficulty})\n"
            f"  - Team Stats (S/P/C): {self.team.total_team_smile}/{self.team.total_team_pure}/{self.team.total_team_cool}\n"
            f"  - Accuracy: {self.accuracy:.2%}\n"
            f"  - Approach Rate: {self.approach_rate}\n"
            f"  - PPN (Points Per Note): {self.slot_ppn_values}"
        )
        return f"{header}\n{details}"
