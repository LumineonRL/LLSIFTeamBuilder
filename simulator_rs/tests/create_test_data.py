import json
import os
import sys
from dataclasses import asdict

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the project root (2 levels up)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
# Add the 'src' directory to the Python path
sys.path.append(os.path.join(project_root, "src"))

from simulator.core.skill import Skill
from simulator.core.leader_skill import LeaderSkill


def main():
    """Creates JSON files for testing the Rust implementation."""
    # Create a sample Skill
    skill = Skill(
        type="score_up",
        activation="notes",
        target="all",
        level=[1, 2, 3, 4, 5, 6, 7, 8],
        thresholds=[20, 25, 30, 35, 40, 45, 50, 55],
        chances=[0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51],
        values=[100, 200, 300, 400, 500, 600, 700, 800],
        durations=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
    )

    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "skill.json"), "w") as f:
        json.dump(asdict(skill), f, indent=4)

    # Create a sample LeaderSkill
    leader_skill = LeaderSkill(
        attribute="smile",
        value=0.09,
        extra_attribute="cool",
        extra_target="bibi",
        extra_value=0.03,
    )

    with open(os.path.join(output_dir, "leader_skill.json"), "w") as f:
        json.dump(asdict(leader_skill), f, indent=4)


if __name__ == "__main__":
    main()
