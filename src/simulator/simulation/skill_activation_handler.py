"""
This module handles the logic of skill activation.

It determines if a skill's trigger conditions are met, checks its activation
probability, and manages the special activation ordering rules
for different skill types (Amplify / Encore / Other). It also manages the state
and trigger conditions for skills like Score-based Scorers or  Year Group skills.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

import src.simulator.simulation.effect_handler as effect_handler
from src.simulator.card.card import Card
from src.simulator.simulation.events import Event, EventType

if TYPE_CHECKING:
    from src.simulator.simulation.play import Play
    from src.simulator.simulation.trial_state import TrialState


class SkillActivationHandler:
    """
    Orchestrates the entire skill activation process for a given trigger.
    """

    def __init__(self, random_state: np.random.Generator, logger: logging.Logger):
        self.random_state = random_state
        self.logger = logger
        self.amp_consumed_this_tick = False

    def _check_score_triggers(
        self,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
    ):
        """Processes skills activated by reaching a score threshold."""
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
            self.process_triggers(
                "Score", triggered, state, play, event_queue, current_time
            )

    def process_triggers(
        self,
        activation_type: str,
        triggered_items: List[Dict],
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
    ):
        """
        Processes a list of items whose skill trigger condition has been met.
        """
        if not triggered_items:
            return

        self.amp_consumed_this_tick = False

        skill_groups = {"Amplify": [], "Encore": [], "Other": []}
        for s in triggered_items:
            skill_type = s["card"].skill.type
            if skill_type == "Amplify":
                skill_groups["Amplify"].append(s)
            elif skill_type == "Encore":
                skill_groups["Encore"].append(s)
            else:
                skill_groups["Other"].append(s)

        for group in skill_groups.values():
            group.sort(key=lambda s: s["slot_idx"], reverse=True)

        order = (
            ["Other", "Amplify", "Encore"]
            if not state.coin_flip
            else ["Amplify", "Other", "Encore"]
        )

        for category_key in order:
            for skill_info in skill_groups[category_key]:
                self._attempt_skill_activation(
                    skill_info["card"],
                    skill_info["slot_idx"],
                    activation_type,
                    state,
                    play,
                    event_queue,
                    current_time,
                )

    def _attempt_skill_activation(
        self,
        card: Card,
        slot_idx: int,
        activation_type: str,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
    ):
        """
        Handles the activation attempt for a single card and its accessory.
        """
        card_activated, amp_used = self._check_rng_and_activate(
            card,
            slot_idx,
            activation_type,
            state,
            play,
            event_queue,
            current_time,
            is_accessory=False,
        )
        if amp_used:
            self.amp_consumed_this_tick = True

        if card_activated:
            return

        accessory = play.team.slots[slot_idx].accessory
        if not accessory:
            self.logger.info(
                "SKILL: (%d) %s's skill failed. No accessory to attempt.",
                slot_idx + 1,
                card.display_name,
            )
            return

        self.logger.info(
            "SKILL: (%d) %s's skill failed. Checking for accessory skill...",
            slot_idx + 1,
            card.display_name,
        )
        acc_activated, amp_used = self._check_rng_and_activate(
            accessory,
            slot_idx,
            "Accessory",
            state,
            play,
            event_queue,
            current_time,
            is_accessory=True,
        )

        if acc_activated and amp_used:
            self.amp_consumed_this_tick = True
        elif not acc_activated:
            self.logger.info(
                "SKILL: Accessory %s on (%d) %s also failed to activate.",
                accessory.name,
                slot_idx + 1,
                card.display_name,
            )

    def _check_rng_and_activate(
        self,
        skilled_item: Any,
        slot_idx: int,
        activation_type: str,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
        is_accessory: bool,
    ) -> Tuple[bool, bool]:
        """
        Checks RNG for a single skill and activates if successful.
        """
        amp_to_use = 0
        if skilled_item.skill.type != "Amplify" and not self.amp_consumed_this_tick:
            amp_to_use = state.active_amp_boost

        eff_lvl = (
            skilled_item.skill_level
            if is_accessory
            else skilled_item.current_skill_level
        ) + amp_to_use

        base_chance = (
            skilled_item.get_skill_attribute_for_level(
                skilled_item.skill.chances, eff_lvl
            )
            or 0.0
        )

        sru_boost = 0.0
        if state.active_sru_effect and slot_idx != state.active_sru_effect["slot_idx"]:
            sru_boost = state.active_sru_effect["value"]

        if self.random_state.random() > (base_chance + sru_boost):
            return False, False

        amp_was_consumed = False
        if amp_to_use > 0:
            state.active_amp_boost = 0
            amp_was_consumed = True

        self._activate_skill(
            skilled_item,
            slot_idx,
            activation_type,
            state,
            play,
            event_queue,
            current_time,
            eff_lvl,
        )
        return True, amp_was_consumed

    def _activate_skill(
        self,
        skilled_item: Any,
        slot_idx: int,
        activation_type: str,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
        eff_lvl: int,
    ):
        # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
        """
        Dispatches to the correct effect handler and logs the detailed result.
        """
        skill_type = skilled_item.skill.type
        item_name = (
            skilled_item.display_name
            if isinstance(skilled_item, Card)
            else skilled_item.name
        )

        match skill_type:
            case "Scorer" | "Healer":
                score = effect_handler.apply_score_effect(
                    state,
                    skilled_item,
                    slot_idx,
                    play.team.slots,
                    play.game_data,
                    eff_lvl,
                )
                self.logger.info(
                    "SKILL: (%d) %s's %s activated for %d points.",
                    slot_idx + 1,
                    item_name,
                    skill_type,
                    score,
                )
                state.last_skill_info = {
                    "item": skilled_item,
                    "slot_index": slot_idx,
                    "type": skill_type,
                    "score_gain": score,
                    "duration": 0,
                }
                if score > 0:
                    self._check_score_triggers(state, play, event_queue, current_time)

            case "Perfect Lock":
                duration = effect_handler.apply_perfect_lock_effect(
                    state,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    skilled_item,
                    eff_lvl,
                    play,
                )
                self.logger.info(
                    "SKILL: (%d) %s's Perfect Lock activated for %.2f seconds.",
                    slot_idx + 1,
                    item_name,
                    duration,
                )
                state.last_skill_info = {
                    "item": skilled_item,
                    "slot_index": slot_idx,
                    "type": skill_type,
                    "score_gain": 0,
                    "duration": duration,
                }
            case "Total Trick":
                duration = effect_handler.apply_total_trick_effect(
                    state, current_time, state.song_end_time, skilled_item, eff_lvl
                )
                self.logger.info(
                    "SKILL: (%d) %s's Total Trick activated for %.2f seconds.",
                    slot_idx + 1,
                    item_name,
                    duration,
                )
                state.last_skill_info = {
                    "item": skilled_item,
                    "slot_index": slot_idx,
                    "type": skill_type,
                    "score_gain": 0,
                    "duration": duration,
                }
            case "Amplify":
                value = (
                    skilled_item.get_skill_attribute_for_level(
                        skilled_item.skill.values, eff_lvl
                    )
                    or 0
                )
                state.active_amp_boost += int(value)
                self.logger.info(
                    "SKILL: (%d) %s's Amplify activated, boosting next skill's level by +%d.",
                    slot_idx + 1,
                    item_name,
                    value,
                )
                state.last_skill_info = {
                    "type": "Amplify",
                    "item": skilled_item,
                    "slot_index": slot_idx,
                }
            case "Encore":
                state.spark_charges += 1
                self.logger.info(
                    "SKILL: (%d) %s's Encore activated. Spark charges are now %d.",
                    slot_idx + 1,
                    item_name,
                    state.spark_charges,
                )
                if state.last_skill_info and state.last_skill_info.get("type") not in [
                    "Encore",
                    "Amplify",
                ]:
                    self._activate_encore_skill(
                        state,
                        play,
                        event_queue,
                        current_time,
                        state.last_skill_info,
                        eff_lvl,
                    )
                else:
                    self.logger.info(
                        "-> Encore triggered but had nothing valid to copy."
                    )
                state.last_skill_info = {"type": "Encore"}
            case "Skill Rate Up":
                if state.active_sru_effect:
                    self.logger.info(
                        f"SKILL: ({slot_idx + 1}) {item_name}'s "
                        "Skill Rate Up triggered, but another is "
                        "already active. No effect."
                    )
                else:
                    duration, boost_val = effect_handler.apply_skill_rate_up_effect(
                        state,
                        skilled_item,
                        slot_idx,
                        current_time,
                        state.song_end_time,
                        event_queue,
                        eff_lvl,
                    )
                    self.logger.info(
                        "SKILL: (%d) %s's Skill Rate Up activated, boosting skill chance by %.2f%% for %.2f seconds.",
                        slot_idx + 1,
                        item_name,
                        boost_val * 100,
                        duration,
                    )
            case "Appeal Boost":
                if state.active_appeal_boost:
                    self.logger.info(
                        f"SKILL: ({slot_idx + 1}) {item_name}'s "
                        "Appeal Boost triggered, but another is "
                        "already active. No effect."
                    )
                else:
                    duration, boost_val, target_str = (
                        effect_handler.apply_appeal_boost_effect(
                            state,
                            skilled_item,
                            slot_idx,
                            play.team.slots,
                            play.game_data,
                            current_time,
                            state.song_end_time,
                            event_queue,
                            play,
                            eff_lvl,
                        )
                    )
                    if target_str:
                        self.logger.info(
                            "SKILL: (%d) %s's Appeal Boost activated, increasing stats of %s by %.2f%% for %.2f seconds.",
                            slot_idx + 1,
                            item_name,
                            target_str,
                            boost_val * 100,
                            duration,
                        )
            case "Sync":
                duration, target_idx, target_name = effect_handler.apply_sync_effect(
                    state,
                    skilled_item,
                    slot_idx,
                    play.team.slots,
                    play.game_data,
                    self.random_state,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    play,
                    eff_lvl,
                )
                if target_idx != -1:
                    self.logger.info(
                        "SKILL: (%d) %s's Sync copies stats from (%d) %s for %.2f seconds.",
                        slot_idx + 1,
                        item_name,
                        target_idx + 1,
                        target_name,
                        duration,
                    )
            case "Perfect Score Up" | "Combo Bonus Up":
                duration = (
                    skilled_item.get_skill_attribute_for_level(
                        skilled_item.skill.durations, eff_lvl
                    )
                    or 0
                )
                value = (
                    skilled_item.get_skill_attribute_for_level(
                        skilled_item.skill.values, eff_lvl
                    )
                    or 0
                )

                effect_list = (
                    state.active_psu_effects
                    if skill_type == "Perfect Score Up"
                    else state.active_cbu_effects
                )
                end_event = (
                    EventType.PERFECT_SCORE_UP_END
                    if skill_type == "Perfect Score Up"
                    else EventType.COMBO_BONUS_UP_END
                )
                log_text = (
                    "points to each Perfect"
                    if skill_type == "Perfect Score Up"
                    else "to combo bonus"
                )

                effect_handler.apply_generic_timed_effect(
                    effect_list,
                    event_queue,
                    end_event,
                    current_time,
                    duration,
                    value,
                    state.song_end_time,
                )
                self.logger.info(
                    "SKILL: (%d) %s's %s activated, adding %d %s for %.2f seconds.",
                    slot_idx + 1,
                    item_name,
                    skill_type,
                    value,
                    log_text,
                    duration,
                )
            case "Spark":
                threshold = (
                    skilled_item.get_skill_attribute_for_level(
                        skilled_item.skill.thresholds, eff_lvl
                    )
                    or 0
                )

                if not threshold or state.spark_charges < threshold:
                    self.logger.info(
                        f"SKILL: ({slot_idx + 1}) {item_name}'s Spark failed to activate. "
                        f"Needs {threshold} charges, has {state.spark_charges}."
                    )
                else:
                    activated, charges, bonus, duration = (
                        effect_handler.apply_spark_effect(
                            state,
                            skilled_item,
                            current_time,
                            state.song_end_time,
                            event_queue,
                            eff_lvl,
                        )
                    )
                    if activated:
                        self.logger.info(
                            f"SKILL: ({slot_idx + 1}) {item_name}'s Spark activated, consuming {charges} charges. "
                            f"Adds {bonus} to tap score for {duration:.2f} seconds. "
                            f"{state.spark_charges} charges remaining."
                        )

        if isinstance(skilled_item, Card):
            self._process_year_group_triggers(
                skilled_item.character, current_time, state, play, event_queue
            )

    def _activate_encore_skill(
        self,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
        current_time: float,
        copied_info: Dict,
        eff_lvl: int,
    ):
        """
        Handles the activation logic for an Encore skill, copying the last one.
        """
        copied_item = copied_info["item"]
        copied_type = copied_info.get("type")
        copied_slot = copied_info.get("slot_index")
        copied_name = (
            copied_item.display_name
            if isinstance(copied_item, Card)
            else copied_item.name
        )

        if copied_slot is None:
            return

        self.logger.info(
            "SKILL: Encore copies %s from (%d) %s.",
            copied_type,
            copied_slot + 1,
            copied_name,
        )

        match copied_type:
            case "Amplify":
                value = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.values, eff_lvl
                    )
                    or 0
                )
                state.active_amp_boost += int(value)
            case "Combo Bonus Up":
                duration = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.durations, eff_lvl
                    )
                    or 0
                )
                value = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.values, eff_lvl
                    )
                    or 0
                )
                effect_handler.apply_generic_timed_effect(
                    state.active_cbu_effects,
                    event_queue,
                    EventType.COMBO_BONUS_UP_END,
                    current_time,
                    duration,
                    value,
                    state.song_end_time,
                )
            case "Perfect Lock":
                effect_handler.apply_perfect_lock_effect(
                    state,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    copied_item,
                    eff_lvl,
                    play,
                )
            case "Total Trick":
                effect_handler.apply_total_trick_effect(
                    state, current_time, state.song_end_time, copied_item, eff_lvl
                )
            case "Perfect Score Up":
                duration = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.durations, eff_lvl
                    )
                    or 0
                )
                value = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.values, eff_lvl
                    )
                    or 0
                )
                effect_handler.apply_generic_timed_effect(
                    state.active_psu_effects,
                    event_queue,
                    EventType.PERFECT_SCORE_UP_END,
                    current_time,
                    duration,
                    value,
                    state.song_end_time,
                )
            case "Appeal Boost":
                effect_handler.apply_appeal_boost_effect(
                    state,
                    copied_item,
                    copied_slot,
                    play.team.slots,
                    play.game_data,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    play,
                    eff_lvl,
                )
            case "Sync":
                effect_handler.apply_sync_effect(
                    state,
                    copied_item,
                    copied_slot,
                    play.team.slots,
                    play.game_data,
                    self.random_state,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    play,
                    eff_lvl,
                )
            case "Skill Rate Up":
                effect_handler.apply_skill_rate_up_effect(
                    state,
                    copied_item,
                    copied_slot,
                    current_time,
                    state.song_end_time,
                    event_queue,
                    eff_lvl,
                )
            case "Spark":
                threshold = (
                    copied_item.get_skill_attribute_for_level(
                        copied_item.skill.thresholds, eff_lvl
                    )
                    or 0
                )
                if not threshold or state.spark_charges < threshold:
                    self.logger.info(
                        "-> Copied Spark skill failed to activate. Needs %d charges, has %d.",
                        threshold,
                        state.spark_charges,
                    )
                else:
                    activated, charges, bonus, duration = (
                        effect_handler.apply_spark_effect(
                            state,
                            copied_item,
                            current_time,
                            state.song_end_time,
                            event_queue,
                            eff_lvl,
                        )
                    )
                    if activated:
                        self.logger.info(
                            "-> Copied Spark skill activated, consuming %d charges. "
                            "Adds %d to tap score for %.2f seconds. %d charges remaining.",
                            charges,
                            bonus,
                            duration,
                            state.spark_charges,
                        )
            case _:  # Handles Scorer/Healer
                if (score_gain := copied_info.get("score_gain", 0)) > 0:
                    state.total_score += score_gain
                    self._check_score_triggers(state, play, event_queue, current_time)

    def _process_year_group_triggers(
        self,
        activating_character: str,
        current_time: float,
        state: "TrialState",
        play: "Play",
        event_queue: List[Event],
    ):
        """
        Processes 'Year Group' skills that require multiple member activations.
        """
        for receiver_idx, required in list(state.year_group_skill_trackers.items()):
            if activating_character in required:
                required.remove(activating_character)

                # Check if all required members have now activated
                if not required:
                    receiver_card = play.team.slots[receiver_idx].card
                    if receiver_card and receiver_card.skill.target:
                        self.logger.info(
                            "SKILL: (%d) %s's Year Group skill is now ready to activate.",
                            receiver_idx + 1,
                            receiver_card.display_name,
                        )
                        # Trigger the skill activation process for the year group card
                        self.process_triggers(
                            "Year Group",
                            [{"card": receiver_card, "slot_idx": receiver_idx}],
                            state,
                            play,
                            event_queue,
                            current_time,
                        )
                        # Reset the tracker for this card for future activations
                        target_group_name = receiver_card.skill.target
                        all_members = play.game_data.sub_group_mapping.get(
                            target_group_name, set()
                        )
                        if receiver_card.character:
                            state.year_group_skill_trackers[receiver_idx] = set(
                                all_members
                            ) - {receiver_card.character}
