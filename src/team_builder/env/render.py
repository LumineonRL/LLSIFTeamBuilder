import sys
from typing import TYPE_CHECKING, List, Dict, Callable

import numpy as np

from src.team_builder.env.build_phase import BuildPhase

if TYPE_CHECKING:
    from src.team_builder.env.env import LLSIFTeamBuildingEnv


def render_human_mode(env: "LLSIFTeamBuildingEnv"):
    """
    Prints a human-readable representation of the environment state.

    Args:
        env: The environment instance to render.
    """
    print("--- Environment State ---")
    phase_name = BuildPhase(env.state.build_phase).name
    print(
        f"Build Phase: {phase_name} ({env.state.build_phase}), "
        f"Current Slot: {env.state.current_slot_idx + 1}"
    )
    if env.reward_mode == "dense":
        print(f"Last Score: {env.state.last_score:.2f}")

    print("\n--- Current Team ---")
    if env.state.team:
        print(repr(env.state.team))
    else:
        print("Team not yet initialized.")

    if env.state.final_approach_rate is not None:
        print("\n--- Final Play ---")
        print(f"Chosen Approach Rate: {env.state.final_approach_rate}")

    print("\n--- Valid Actions ---")
    valid_actions = np.where(env.action_masks())[0]
    print(f"Number of valid actions: {len(valid_actions)}")
    if len(valid_actions) > 20:
        print(
            f"[{valid_actions[0]}, {valid_actions[1]}, ..., "
            f"{valid_actions[-2]}, {valid_actions[-1]}]"
        )
    else:
        print(valid_actions)
    print("-----------------------\n")
    sys.stdout.flush()


def get_agent_render_data(env: "LLSIFTeamBuildingEnv") -> List:
    """
    Returns data structures for the 'agent' render mode based on the
    current build phase.
    """

    def _render_deck_phase() -> List:
        if env.state.team is None:
            return []
        all_ids = set(env.deck.entries.keys())
        unassigned = sorted(list(all_ids - env.state.team.assigned_deck_ids))
        return [env.deck.get_entry(did) for did in unassigned]

    def _render_accessory_phase() -> List:
        if env.state.team is None:
            return ["Pass"]
        return ["Pass"] + env.accessory_manager.get_unassigned_accessories(
            env.state.team.assigned_accessory_ids
        )

    def _render_sis_phase() -> List:
        if env.state.team is None:
            return ["Pass"]
        return ["Pass"] + env.sis_manager.get_unassigned_sis(
            env.state.team.assigned_sis_ids
        )

    def _render_guest_phase() -> List:
        if env.enable_guests and env.guest_manager:
            return list(env.guest_manager.all_guests.values())
        return ["No Guests Enabled"]

    def _render_approach_rate_phase() -> List:
        return list(range(1, env.obs_manager.APPROACH_RATE_CHOICES + 1))

    render_phases: Dict[BuildPhase, Callable[[], List]] = {
        BuildPhase.CARD_SELECTION: _render_deck_phase,
        BuildPhase.ACCESSORY_ASSIGNMENT: _render_accessory_phase,
        BuildPhase.SIS_ASSIGNMENT: _render_sis_phase,
        BuildPhase.GUEST_SELECTION: _render_guest_phase,
        BuildPhase.SCORE_SIMULATION: _render_approach_rate_phase,
    }
    phase = render_phases.get(env.state.build_phase)
    return phase() if phase else []
