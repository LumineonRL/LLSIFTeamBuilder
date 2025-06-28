"""
This module contains handlers for applying the effects of activated skills.

These functions are called by the SkillActivationHandler upon a successful
skill trigger. They are responsible for modifying the trial's state and
scheduling end events for any duration-based effects.
"""

import math
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from src.simulator.card.card import Card
from src.simulator.simulation.events import Event, EventType

if TYPE_CHECKING:
    from src.simulator.simulation.play import Play
    from src.simulator.simulation.trial_state import TrialState


def recalculate_stats_and_ppn(state: "TrialState", play: "Play"):
    """
    Recalculates all slot stats and PPN values based on active effects.

    This is a core function called whenever a stat-modifying effect
    starts or ends.
    """
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
        target_sync_idx = sync_info["target_slot_index"]
        current_stats[slot_idx] = dict(current_stats[target_sync_idx])

    is_trick_active = state.active_pl_count > 0 or state.total_trick_end_time > 0
    if is_trick_active:
        for slot_idx, tricks in play.trick_slots.items():
            for trick_sis in tricks:
                attr = trick_sis.attribute.lower()
                bonus = math.ceil(original_slot_stats[slot_idx][attr] * trick_sis.value)
                current_stats[slot_idx][attr] += bonus

    final_team_stats = {
        "Smile": sum(s.get("smile", 0) for s in current_stats),
        "Pure": sum(s.get("pure", 0) for s in current_stats),
        "Cool": sum(s.get("cool", 0) for s in current_stats),
    }
    state.current_slot_ppn = play.calculate_ppn_for_all_slots(
        final_team_stats.get(play.song.attribute, 0)
    )


def apply_score_effect(
    state: "TrialState",
    skilled_item: Any,
    slot_index: int,
    team_slots: List[Any],
    game_data: Any,
    effective_skill_level: int,
) -> int:
    """
    Calculates the score gain from a Scorer or Healer skill.
    """
    skill_type = skilled_item.skill.type
    value = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.values, effective_skill_level
        )
        or 0
    )
    score_gain = 0

    if skill_type == "Scorer":
        charm_mult = (
            sum(
                s.sis.value
                for s in team_slots[slot_index].sis_entries
                if s.sis.effect == "charm"
            )
            or 1.0
        )
        score_gain = int(value * charm_mult)
    elif skill_type == "Healer":
        has_heal_sis = any(
            s.sis.effect == "heal" for s in team_slots[slot_index].sis_entries
        )
        if has_heal_sis:
            score_gain = int(value * game_data.HEAL_MULTIPLIER)

    if score_gain > 0:
        state.total_score += score_gain

    return score_gain


def apply_perfect_lock_effect(
    state: "TrialState",
    current_time: float,
    song_end_time: float,
    event_queue: List[Event],
    skilled_item: Any,
    effective_skill_level: int,
    play: "Play",
) -> float:
    """
    Applies a Perfect Lock effect, updating state and scheduling the end.
    """
    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )

    if state.active_pl_count == 0 and state.pl_uptime_start_time is None:
        state.pl_uptime_start_time = current_time

    state.active_pl_count += 1
    end_time = min(current_time + duration, song_end_time)

    event_queue.append(Event(end_time, EventType.LOCK_END, payload={"type": "pl_end"}))
    event_queue.sort()

    # Recalculate stats immediately to apply Trick SIS bonus
    recalculate_stats_and_ppn(state, play)

    return duration


def end_perfect_lock_effect(
    state: "TrialState", current_time: float, song_end_time: float
):
    """
    Handles the logic for when a Perfect Lock effect expires.
    """
    state.active_pl_count -= 1
    if state.active_pl_count == 0 and state.pl_uptime_start_time is not None:
        interval = (
            state.pl_uptime_start_time,
            min(current_time, song_end_time),
        )
        state.uptime_intervals.append(interval)
        state.pl_uptime_start_time = None


def apply_total_trick_effect(
    state: "TrialState",
    current_time: float,
    song_end_time: float,
    skilled_item: Any,
    effective_skill_level: int,
) -> float:
    """
    Applies a Total Trick effect by extending its end time.
    """
    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )
    state.total_trick_end_time = max(
        state.total_trick_end_time,
        min(current_time + duration, song_end_time),
    )
    return duration


def apply_generic_timed_effect(
    state_effects_list: List[Dict],
    event_queue: List[Event],
    event_type: EventType,
    current_time: float,
    duration: float,
    value: Any,
    song_end_time: float,
) -> uuid.UUID:
    """
    A generic handler for simple timed effects like PSU and CBU.
    """
    effect_id = uuid.uuid4()
    end_time = min(current_time + duration, song_end_time)

    state_effects_list.append({"id": effect_id, "value": value})
    event_queue.append(Event(end_time, event_type, payload={"id": effect_id}))
    event_queue.sort()
    return effect_id


def end_generic_timed_effect(
    state_effects_list: List[Dict], effect_id: uuid.UUID
) -> bool:
    """
    Removes an effect from the state's active list by its unique ID.
    """
    initial_count = len(state_effects_list)
    state_effects_list[:] = [
        eff for eff in state_effects_list if eff.get("id") != effect_id
    ]
    return len(state_effects_list) < initial_count


def apply_appeal_boost_effect(
    state: "TrialState",
    skilled_item: Any,
    slot_idx: int,
    team_slots: List[Any],
    game_data: Any,
    current_time: float,
    song_end_time: float,
    event_queue: List[Event],
    play: "Play",
    effective_skill_level: int,
) -> Tuple[float, float, str]:
    """
    Applies an Appeal Boost effect and returns details for logging.
    """
    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )
    boost_val = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.values, effective_skill_level
        )
        or 0
    )

    target_group = skilled_item.skill.target
    if target_group is None:
        return duration, boost_val, ""

    target_slots, target_str = set(), ""
    if target_group == "":
        target_slots.add(slot_idx)
        target_str = "self"
    else:
        valid_members = game_data.group_mapping.get(target_group, set())
        target_slots = {
            i
            for i, s in enumerate(team_slots)
            if s.card and s.card.character in valid_members
        }
        target_str = target_group

    if not target_slots:
        return duration, boost_val, ""

    end_time = min(current_time + duration, song_end_time)
    item_name = (
        skilled_item.display_name
        if isinstance(skilled_item, Card)
        else skilled_item.name
    )

    state.active_appeal_boost = {
        "value": boost_val,
        "target_slots": target_slots,
    }
    event_queue.append(
        Event(
            end_time,
            EventType.APPEAL_BOOST_END,
            payload={"item_name": item_name, "slot_idx": slot_idx},
        )
    )
    event_queue.sort()
    recalculate_stats_and_ppn(state, play)
    return duration, boost_val, target_str


def apply_sync_effect(
    state: "TrialState",
    skilled_item: Any,
    slot_idx: int,
    team_slots: List[Any],
    game_data: Any,
    random_state: np.random.Generator,
    current_time: float,
    song_end_time: float,
    event_queue: List[Event],
    play: "Play",
    effective_skill_level: int,
) -> Tuple[float, int, str]:
    """
    Applies a Sync effect and returns details for logging.
    """
    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )
    target_group = skilled_item.skill.target
    if not target_group:
        return duration, -1, ""

    valid_members = game_data.sub_group_mapping.get(target_group, set())
    potential_targets = [
        idx
        for idx, s in enumerate(team_slots)
        if idx != slot_idx and s.card and s.card.character in valid_members
    ]
    if not potential_targets:
        return duration, -1, ""

    target_idx = random_state.choice(potential_targets)
    end_time = min(current_time + duration, song_end_time)

    state.active_sync_effects[slot_idx] = {"target_slot_index": target_idx}
    event_queue.append(
        Event(end_time, EventType.SYNC_END, payload={"slot_idx": slot_idx})
    )
    event_queue.sort()
    recalculate_stats_and_ppn(state, play)

    target_card_name = ""
    if team_slots[target_idx].card:
        target_card_name = team_slots[target_idx].card.display_name

    return duration, target_idx, target_card_name


def apply_skill_rate_up_effect(
    state: "TrialState",
    skilled_item: Any,
    slot_idx: int,
    current_time: float,
    song_end_time: float,
    event_queue: List[Event],
    effective_skill_level: int,
) -> Tuple[float, float]:
    """
    Applies a Skill Rate Up (SRU) effect and returns details for logging.
    """
    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )
    boost_val = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.values, effective_skill_level
        )
        or 0
    )

    end_time = min(current_time + duration, song_end_time)
    item_name = (
        skilled_item.display_name
        if isinstance(skilled_item, Card)
        else skilled_item.name
    )

    state.active_sru_effect = {"value": boost_val, "slot_idx": slot_idx}
    event_queue.append(
        Event(
            end_time,
            EventType.SKILL_RATE_UP_END,
            payload={"item_name": item_name, "slot_idx": slot_idx},
        )
    )
    event_queue.sort()
    return duration, boost_val


def apply_spark_effect(
    state: "TrialState",
    skilled_item: Any,
    current_time: float,
    song_end_time: float,
    event_queue: List[Event],
    effective_skill_level: int,
) -> Tuple[bool, int, int, float]:
    """
    Applies a Spark effect and returns details for logging.
    """
    threshold = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.thresholds, effective_skill_level
        )
        or 0
    )
    if not threshold or state.spark_charges < threshold:
        return False, 0, 0, 0

    duration = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.durations, effective_skill_level
        )
        or 0
    )
    value = (
        skilled_item.get_skill_attribute_for_level(
            skilled_item.skill.values, effective_skill_level
        )
        or 0
    )

    multiplier = math.floor(state.spark_charges / threshold)
    charges_to_consume = multiplier * threshold
    bonus_per_note = multiplier * value

    state.spark_charges -= charges_to_consume

    apply_generic_timed_effect(
        state.active_spark_effects,
        event_queue,
        EventType.SPARK_END,
        current_time,
        duration,
        bonus_per_note,
        song_end_time,
    )
    return True, charges_to_consume, bonus_per_note, duration
