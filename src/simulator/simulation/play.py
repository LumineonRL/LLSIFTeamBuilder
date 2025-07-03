"""
This module defines the main Play class, which orchestrates a simulation
of a team playing a song.
"""

# pylint: disable=too-few-public-methods

import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.simulator.card.card import Card
from src.simulator.simulation.game_data import GameData
from src.simulator.simulation.play_config import PlayConfig
from src.simulator.simulation.trial import Trial
from src.simulator.sis.sis import SIS
from src.simulator.song.note import Note
from src.simulator.song.song import Song
from src.simulator.team.team import Team


class Play:
    """
    Represents an instance of a song being played by a specific team.

    This class orchestrates the simulation but delegates the heavy lifting
    of a single run to the `Trial` class. It is responsible for setting up
    the simulation environment, running trials, and reporting results.
    """

    PPN_BASE_FACTOR = 0.0125
    GROUP_BONUS = 0.1
    ATTRIBUTE_BONUS = 0.1

    NOTE_MULTIPLIERS = {
        "is_hold_and_swing": 0.625,
        "is_hold": 1.25,
        "is_swing": 0.5,
        "default": 1.0,
    }

    def __init__(self, team: Team, song: Song, config: PlayConfig, game_data: GameData):
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
        self.logger: logging.Logger | None = None

        # Pre-calculate values that are constant across all trials
        self.trick_slots: Dict[int, List[SIS]] = {
            i: [s.sis for s in slot.sis_entries if s.sis.effect == "trick"]
            for i, slot in enumerate(self.team.slots)
            if slot.card
        }
        team_total_stat = getattr(
            self.team, f"total_team_{self.song.attribute.lower()}", 0
        )
        self.base_slot_ppn: List[int] = self.calculate_ppn_for_all_slots(
            team_total_stat
        )

    def simulate(self, n_trials: int = 1, log_level: Optional[int] = None) -> List[int]:
        """
        Runs the simulation for a specified number of trials.

        Args:
            n_trials: The number of trials to run.
            log_level: The logging level to use (e.g., logging.INFO).
                    If None, uses the level from PlayConfig.

        Returns:
            A list of the final integer scores for each trial.
        """
        effective_log_level = (
            log_level if log_level is not None else self.config.log_level
        )
        self.logger = self._setup_logger(effective_log_level)
        trial_scores: List[int] = []
        trial_uptimes: List[float] = []

        self.logger.debug("--- Starting Simulation for %s ---", self)
        for i in range(n_trials):
            self.logger.debug(
                "--- Starting Simulation Trial %d/%d ---", i + 1, n_trials
            )

            trial = Trial(self, self.random_state)
            trial.run()

            trial_scores.append(trial.total_score)
            total_uptime = trial.get_total_pl_uptime()
            trial_uptimes.append(total_uptime)
            self._log_trial_summary(trial, total_uptime)

            self.logger.debug(
                "--- Trial %d Finished. Final Score: %s ---",
                i + 1,
                f"{trial.total_score:,}",
            )

        if n_trials > 1:
            self._log_overall_summary(trial_scores, trial_uptimes)

        return trial_scores

    # --- PPN and Multiplier Calculation Helpers ---

    def _check_group_bonus(self, card: Card) -> float:
        """Checks if a card's group matches the song's for a bonus."""
        song_group = self.song.group
        valid_members = self.game_data.group_mapping.get(song_group, set())
        return self.GROUP_BONUS if card.character in valid_members else 0.0

    def _check_attribute_bonus(self, card: Card) -> float:
        """Checks if a card's attribute matches the song's for a bonus."""
        return self.ATTRIBUTE_BONUS if self.song.attribute == card.attribute else 0.0

    def calculate_ppn_for_all_slots(self, team_total_stat: int) -> List[int]:
        """Calculates the PPN for each team slot given a total team stat."""
        if team_total_stat == 0:
            warnings.warn("Team total stat for song attribute is 0. All PPN will be 0.")
            return [0] * self.team.NUM_SLOTS

        ppn_values = []
        for slot in self.team.slots:
            if not slot.card:
                ppn_values.append(0)
                continue

            group_bonus = self._check_group_bonus(slot.card)
            attribute_bonus = self._check_attribute_bonus(slot.card)

            total_bonus = 1 + group_bonus + attribute_bonus
            slot_ppn = math.floor(team_total_stat * self.PPN_BASE_FACTOR * total_bonus)
            ppn_values.append(slot_ppn)
        return ppn_values

    @staticmethod
    def get_note_multiplier(note: Note) -> float:
        """Determines the score multiplier for a given note type."""
        is_hold = note.start_time != note.end_time
        if is_hold and note.is_swing:
            return Play.NOTE_MULTIPLIERS["is_hold_and_swing"]
        if is_hold:
            return Play.NOTE_MULTIPLIERS["is_hold"]
        if note.is_swing:
            return Play.NOTE_MULTIPLIERS["is_swing"]
        return Play.NOTE_MULTIPLIERS["default"]

    @staticmethod
    def get_combo_multiplier(combo_count: int, game_data: GameData) -> float:
        """Finds the combo multiplier for the current combo count."""
        for threshold, multiplier in game_data.combo_bonus_tiers:
            if combo_count + 1 >= threshold:
                return multiplier
        return 1.0  # Default if no tier is met

    # --- Logging and Output ---

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Configures a logger to write simulation results to a file."""
        logger = logging.getLogger("simulation_logger")

        if not self.config.enable_logging:
            if logger.hasHandlers():
                logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.CRITICAL + 1)
            return logger

        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        sanitized_title = "".join(
            c for c in self.song.title if c.isalnum() or c in " _"
        ).rstrip()
        timestamp = int(time.time())
        log_filename = (
            f"{sanitized_title.replace(' ', '_')}_"
            f"{self.song.difficulty}_{timestamp}.log"
        )
        log_filepath = log_dir / log_filename

        logger.setLevel(log_level)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        return logger

    def _log_trial_summary(self, trial: Trial, total_uptime: float):
        """Logs the summary of a single completed trial."""
        if not self.logger:
            return

        self.logger.debug("--- Trial Summary ---")
        self.logger.debug("Final Score: %s", f"{trial.total_score:,}")

        hold_starts = len(trial.hold_note_start_results)
        total_judgements = trial.notes_hit + hold_starts

        ratio_percent = (
            (trial.perfect_hits / total_judgements * 100) if total_judgements > 0 else 0
        )
        uptime_percent = (
            (total_uptime / self.song.length * 100) if self.song.length > 0 else 0
        )

        self.logger.debug(
            "Perfect Ratio: %d / %d (%.2f%%)",
            trial.perfect_hits,
            total_judgements,
            ratio_percent,
        )
        self.logger.debug(
            "Perfect Lock Uptime: %.2fs (%.2f%%)", total_uptime, uptime_percent
        )

    def _log_overall_summary(self, scores: List[int], uptimes: List[float]):
        """Logs the summary of all completed trials."""
        if not self.logger:
            return

        self.logger.debug("\n--- Overall Simulation Summary ---")
        self.logger.debug("Trials Run: %d", len(scores))
        self.logger.debug("Average Score: %s", f"{np.mean(scores):,.0f}")
        self.logger.debug("Max Score: %s", f"{np.max(scores):,}")
        self.logger.debug("Min Score: %s", f"{np.min(scores):,}")
        self.logger.debug("Standard Deviation: %.2f", np.std(scores))

        avg_uptime = np.mean(uptimes)
        avg_uptime_percent = (
            (avg_uptime / self.song.length * 100) if self.song.length > 0 else 0
        )

        self.logger.debug(
            "Average Perfect Lock Uptime: %.2fs (%.2f%%)",
            avg_uptime,
            avg_uptime_percent,
        )

    def __repr__(self) -> str:
        """Provides a summary of the Play configuration."""
        team_stats = (
            f"{self.team.total_team_smile}/"
            f"{self.team.total_team_pure}/"
            f"{self.team.total_team_cool}"
        )
        return (
            f"<Play(Song='{self.song.title}' [{self.song.difficulty}], "
            f"Team Stats='{team_stats}', "
            f"Accuracy={self.config.accuracy:.2%})>"
        )
