import sys
from typing import TYPE_CHECKING

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

    # Access dynamic state attributes via env.state
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
