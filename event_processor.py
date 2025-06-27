"""
This module processes events from the main simulation queue.

The EventProcessor class acts as a dispatcher, taking an event and routing it
to the appropriate logic for handling state changes, scoring, and skill
activations.
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List

from events import Event, EventType
from note import Note
import effect_handler
from skill_activation_handler import SkillActivationHandler

if TYPE_CHECKING:
    from play import Play
    from trial_state import TrialState


class EventProcessor:
    """Dispatches events to appropriate handlers."""

    def __init__(self, play_context: Dict[str, Any]):
        self.game_data = play_context["game_data"]
        self.logger = play_context["logger"]
        self.skill_handler = SkillActivationHandler(
            play_context["random_state"], self.logger
        )

    def dispatch(
        self, event: Event, state: "TrialState", play: "Play", event_queue: List[Event]
    ):
        """Routes an event to its corresponding handler method."""
        handlers = {
            EventType.NOTE_SPAWN: self._handle_note_spawn,
            EventType.TIME_SKILL: self._handle_time_skill,
            EventType.LOCK_END: self._handle_lock_end,
            EventType.SYNC_END: self._handle_sync_end,
            EventType.SKILL_RATE_UP_END: self._handle_sru_end,
            EventType.APPEAL_BOOST_END: self._handle_appeal_boost_end,
            EventType.PERFECT_SCORE_UP_END: self._handle_psu_end,
            EventType.COMBO_BONUS_UP_END: self._handle_cbu_end,
            EventType.SPARK_END: self._handle_spark_end,
            EventType.NOTE_START: self._handle_note_start,
            EventType.NOTE_COMPLETION: self._handle_note_completion,
            EventType.SONG_END: self._handle_song_end,
        }
        handler = handlers.get(event.priority)
        if handler:
            context = {"state": state, "play": play, "event_queue": event_queue}
            handler(event, context)

    # --- Event Handlers ---

    def _handle_note_spawn(self, event: Event, context: Dict):
        """Handles the appearance of a note icon on screen."""
        state = context["state"]
        self.logger.info(
            "EVENT @ %.3fs: Note #%d %s spawns.",
            event.time,
            event.payload["note_idx"] + 1,
            event.payload["spawn_type"],
        )
        state.spawn_events_processed += 1
        self._process_counter_skill(
            "Rhythm Icons", state.spawn_events_processed, event.time, context
        )

    def _handle_time_skill(self, event: Event, context: Dict):
        """Handles a time-based skill activation check."""
        self.skill_handler.process_triggers(
            "Time",
            [event.payload],
            context["state"],
            context["play"],
            context["event_queue"],
            event.time,
        )

    def _handle_lock_end(self, event: Event, context: Dict):
        """Handles the end of a Perfect Lock or Total Trick effect."""
        state = context["state"]
        play = context["play"]
        if event.payload.get("type") == "pl_end":
            self.logger.info("EVENT @ %.3fs: A Perfect Lock effect ended.", event.time)
            effect_handler.end_perfect_lock_effect(state, event.time, play.song.length)
            self._update_stats_and_ppn(context)

    def _handle_sync_end(self, event: Event, context: Dict):
        """Handles the end of a Sync skill effect."""
        state = context["state"]
        slot_idx = event.payload["slot_idx"]
        if slot_idx in state.active_sync_effects:
            card = context["play"].team.slots[slot_idx].card
            if card:
                self.logger.info(
                    "EVENT @ %.3fs: Sync effect ended for (%d) %s.",
                    event.time,
                    slot_idx + 1,
                    card.display_name,
                )
            del state.active_sync_effects[slot_idx]
            self._update_stats_and_ppn(context)

    def _handle_note_start(self, event: Event, context: Dict):
        """Handles the start judgement for a hold note."""
        state = context["state"]
        play = context["play"]
        note_idx = event.payload["note_idx"]

        self.logger.info(
            "EVENT @ %.3fs: Processing hold note start for Note #%d.",
            event.time,
            note_idx + 1,
        )
        if play.random_state.random() <= play.config.accuracy:
            state.hold_note_start_results[note_idx] = "Perfect"
            state.perfect_hits += 1
            self._process_counter_skill(
                "Perfects", state.perfect_hits, event.time, context
            )
        else:
            state.hold_note_start_results[note_idx] = "Great"

    def _handle_note_completion(self, event: Event, context: Dict):
        # pylint: disable=too-many-locals, too-many-branches
        """Handles the scoring and combo update for a completed note."""
        state = context["state"]
        play = context["play"]
        note_idx, current_time = event.payload["note_idx"], event.time
        note: Note = play.song.notes[note_idx]

        is_hold = note.start_time != note.end_time
        note_type_str = "regular"
        if is_hold and note.is_swing:
            note_type_str = "swing hold"
        elif is_hold:
            note_type_str = "hold"
        elif note.is_swing:
            note_type_str = "swing"

        original_hit_is_perfect = play.random_state.random() <= play.config.accuracy
        start_hit_is_perfect = state.hold_note_start_results.get(note_idx) != "Great"
        final_original_perfect = original_hit_is_perfect and (
            not is_hold or start_hit_is_perfect
        )

        is_pl_active = (
            state.active_pl_count > 0 or current_time <= state.total_trick_end_time
        )
        hit_type = "Perfect" if is_pl_active or final_original_perfect else "Great"

        if hit_type == "Perfect":
            state.perfect_hits += 1
            self._process_counter_skill(
                "Perfects", state.perfect_hits, current_time, context
            )

        if is_pl_active and final_original_perfect:
            accuracy_multiplier = 1.08
        elif hit_type == "Perfect":
            accuracy_multiplier = 1.0
        else:
            accuracy_multiplier = 0.88

        hitting_slot_index = note.position - 1
        if 0 <= hitting_slot_index < len(play.team.slots):
            base_ppn = state.current_slot_ppn[hitting_slot_index]
            note_mult = play.get_note_multiplier(note)
            combo_mult = play.get_combo_multiplier(state.combo_count, self.game_data)
            note_score = math.floor(
                base_ppn * note_mult * combo_mult * accuracy_multiplier
            )
            if hit_type == "Perfect" and state.active_psu_effects:
                note_score += sum(eff["value"] for eff in state.active_psu_effects)
            if state.active_cbu_effects:
                cbu_multiplier = next(
                    (
                        m
                        for t, m in self.game_data.combo_fever_map
                        if state.combo_count + 1 >= t
                    ),
                    1.0,
                )
                bonus = sum(
                    math.floor(eff["value"] * cbu_multiplier)
                    for eff in state.active_cbu_effects
                )
                note_score += min(bonus, self.game_data.MAX_COMBO_FEVER_BONUS)

            if state.active_spark_effects:
                spark_bonus = sum(eff["value"] for eff in state.active_spark_effects)
                state.total_score += spark_bonus

            state.total_score += note_score
            slot_card = play.team.slots[hitting_slot_index].card
            if self.logger and slot_card:
                self.logger.info(
                    "(%d) %s hit a %s on %s note #%d for %d points at %.3fs.",
                    hitting_slot_index + 1,
                    slot_card.display_name,
                    hit_type,
                    note_type_str,
                    note_idx + 1,
                    note_score,
                    current_time,
                )

        state.notes_hit += 1
        state.combo_count += 1
        self._process_counter_skill("Combo", state.combo_count, current_time, context)
        if note.is_star:
            self._process_star_note_triggers(current_time, context)
        self._process_score_triggers(current_time, context)

    def _handle_song_end(self, event: Event, context: Dict):
        """Handles the end of the song."""
        self.logger.info(
            "EVENT @ %.3fs: Song has officially ended.",
            event.time,
        )
        context["state"].song_has_ended = True

    # --- Skill Trigger Helpers ---

    def _process_counter_skill(
        self, activation_type: str, counter: int, current_time: float, context: Dict
    ):
        """Checks and triggers skills based on a counter."""
        triggered = []
        for i, slot in enumerate(context["play"].team.slots):
            card = slot.card
            if not (card and card.skill.activation == activation_type):
                continue

            threshold = card.skill_threshold
            if threshold and counter > 0 and counter % threshold == 0:
                triggered.append({"card": card, "slot_idx": i})

        if triggered:
            self.skill_handler.process_triggers(
                activation_type,
                triggered,
                context["state"],
                context["play"],
                context["event_queue"],
                current_time,
            )

    def _process_star_note_triggers(self, current_time: float, context: Dict):
        """Processes skills activated by Star Notes."""
        triggered = [
            {"card": s.card, "slot_idx": i}
            for i, s in enumerate(context["play"].team.slots)
            if s.card and s.card.skill.activation == "Star Notes"
        ]
        if triggered:
            self.skill_handler.process_triggers(
                "Star Notes",
                triggered,
                context["state"],
                context["play"],
                context["event_queue"],
                current_time,
            )

    def _process_score_triggers(self, current_time: float, context: Dict):
        """Processes skills activated by reaching a score threshold."""
        state = context["state"]
        play = context["play"]
        triggered = []
        for idx, slot in enumerate(play.team.slots):
            card = slot.card
            if not (card and card.skill.activation == "Score" and card.skill_threshold):
                continue

            next_thresh = state.score_skill_trackers.get(idx, card.skill_threshold)
            if state.total_score >= next_thresh:
                if card.skill_threshold:
                    state.score_skill_trackers[idx] = next_thresh + card.skill_threshold
                triggered.append({"card": card, "slot_idx": idx})

        if triggered:
            self.skill_handler.process_triggers(
                "Score", triggered, state, play, context["event_queue"], current_time
            )

    def _update_stats_and_ppn(self, context: Dict):
        """
        Recalculates all slot stats and PPN values based on active effects.
        """
        state = context["state"]
        play = context["play"]

        original_slot_stats = [
            {"smile": s.total_smile, "pure": s.total_pure, "cool": s.total_cool}
            for s in play.team.slots
        ]
        current_stats = [dict(s) for s in original_slot_stats]

        if state.active_appeal_boost:
            boost_val = state.active_appeal_boost["value"]
            for target_idx in state.active_appeal_boost["target_slots"]:
                for attr in ["smile", "pure", "cool"]:
                    current_stats[target_idx][attr] = math.ceil(
                        current_stats[target_idx][attr] * (1 + boost_val)
                    )

        for slot_idx, sync_info in state.active_sync_effects.items():
            current_stats[slot_idx] = dict(
                current_stats[sync_info["target_slot_index"]]
            )

        is_trick_active = state.active_pl_count > 0 or state.total_trick_end_time > 0
        if is_trick_active:
            for slot_idx, tricks in play.trick_slots.items():
                for trick_sis in tricks:
                    attr = trick_sis.attribute.lower()
                    bonus = math.ceil(
                        original_slot_stats[slot_idx][attr] * trick_sis.value
                    )
                    current_stats[slot_idx][attr] += bonus

        final_team_stats = {
            "Smile": sum(s.get("smile", 0) for s in current_stats),
            "Pure": sum(s.get("pure", 0) for s in current_stats),
            "Cool": sum(s.get("cool", 0) for s in current_stats),
        }
        state.current_slot_ppn = play.calculate_ppn_for_all_slots(
            final_team_stats.get(play.song.attribute, 0)
        )

    # --- Effect End Handlers ---
    def _handle_sru_end(self, event: Event, context: Dict):
        """Handles the end of a Skill Rate Up effect."""
        self.logger.info(
            "EVENT @ %.3fs: Skill Rate Up effect from (%d) %s ended.",
            event.time,
            event.payload["slot_idx"] + 1,
            event.payload["item_name"],
        )
        context["state"].active_sru_effect = None

    def _handle_appeal_boost_end(self, event: Event, context: Dict):
        """Handles the end of an Appeal Boost effect."""
        self.logger.info(
            "EVENT @ %.3fs: Appeal Boost effect from (%d) %s ended.",
            event.time,
            event.payload["slot_idx"] + 1,
            event.payload["item_name"],
        )
        context["state"].active_appeal_boost = None
        self._update_stats_and_ppn(context)

    def _handle_psu_end(self, event: Event, context: Dict):
        """Handles the end of a Perfect Score Up effect."""
        if effect_handler.end_generic_timed_effect(
            context["state"].active_psu_effects, event.payload.get("id")
        ):
            self.logger.info(
                "EVENT @ %.3fs: A Perfect Score Up effect has ended.", event.time
            )

    def _handle_cbu_end(self, event: Event, context: Dict):
        """Handles the end of a Combo Bonus Up effect."""
        if effect_handler.end_generic_timed_effect(
            context["state"].active_cbu_effects, event.payload.get("id")
        ):
            self.logger.info(
                "EVENT @ %.3fs: A Combo Bonus Up effect has ended.", event.time
            )

    def _handle_spark_end(self, event: Event, context: Dict):
        """Handles the end of a Spark tap score bonus effect."""
        if effect_handler.end_generic_timed_effect(
            context["state"].active_spark_effects, event.payload.get("id")
        ):
            self.logger.info(
                "EVENT @ %.3fs: A Spark tap score bonus has ended.", event.time
            )
