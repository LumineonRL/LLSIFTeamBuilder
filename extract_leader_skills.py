import json
import sys
from pathlib import Path
from typing import List, Tuple, Any, Dict

def process_leader_skills(input_path_str: str, output_path_str: str) -> None:
    """
    Parses a JSON file of cards, extracts unique leader skills, sorts them,
    assigns a sequential ID, and writes the result to a new JSON file.

    Used for Guests.

    Args:
        input_path_str (str): The path to the source JSON file of cards.
        output_path_str (str): The path where the output JSON file will be saved.
    """
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    unique_skills_set = set()

    try:
        with input_path.open('r', encoding='utf-8') as f:
            all_cards = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file not found at {input_path}\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Input file at {input_path} contains invalid JSON.\n")
        sys.exit(1)

    # Find all unique combinations of skill attributes
    for card in all_cards:
        leader_skill = card.get('leader_skill')
        if not leader_skill:
            continue

        extra_skill = leader_skill.get('extra', {}) or {}

        skill_tuple = (
            leader_skill.get('leader_attribute'),
            leader_skill.get('leader_secondary_attribute'),
            leader_skill.get('leader_value'),
            extra_skill.get('leader_extra_attribute'),
            extra_skill.get('leader_extra_target'),
            extra_skill.get('leader_extra_value'),
        )
        unique_skills_set.add(skill_tuple)

    # Custom sort  for each leader skill
    ATTRIBUTE_ORDER = {None: 0, "Smile": 1, "Pure": 2, "Cool": 3}
    sorted_skills_tuples: List[Tuple[Any, ...]] = sorted(
        list(unique_skills_set),
        key=lambda skill: (
            ATTRIBUTE_ORDER.get(skill[0], 99),
            ATTRIBUTE_ORDER.get(skill[1], 99),
            (skill[2] is not None, skill[2])
        )
    )

    output_data: List[Dict[str, Any]] = []
    # Enumerate starting from 1 to generate sequential IDs for the sorted skills
    for i, skill_tuple in enumerate(sorted_skills_tuples, start=1):
        skill_dict = {
            "leader_skill_id": i,
            "leader_attribute": skill_tuple[0],
            "leader_secondary_attribute": skill_tuple[1],
            "leader_value": skill_tuple[2],
            "extra": {
                "leader_extra_attribute": skill_tuple[3],
                "leader_extra_target": skill_tuple[4],
                "leader_extra_value": skill_tuple[5],
            }
        }
        output_data.append(skill_dict)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully wrote {len(output_data)} unique leader skills to {output_path}")
    except IOError as e:
        sys.stderr.write(f"Error: Could not write to file at {output_path}. Reason: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    CARDS_INPUT_PATH = './data/cards.json'
    SKILLS_OUTPUT_PATH = './data/unique_leader_skills.json'
    process_leader_skills(CARDS_INPUT_PATH, SKILLS_OUTPUT_PATH)
