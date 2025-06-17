import sqlite3
import json
import os
from typing import List, Dict, Any

DATA_DIR = './data'
DATABASE_PATH = os.path.join(DATA_DIR, 'unit')
CHARACTERS_JSON_PATH = os.path.join(DATA_DIR, 'characters.json')
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, 'sis.json')

def load_character_map(path: str) -> Dict[str, str]:
    """Loads the characters.json file which is a direct mapping of character ID to name."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            character_map = json.load(f)
        return character_map
    except FileNotFoundError:
        print(f"Error: Character data file not found at '{path}'.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{path}'.")
        return {}

def fetch_skill_records(db_path: str) -> List[sqlite3.Row]:
    """Connects to the SQLite database and fetches all filtered skill records."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Filter out Live Arena SISs. That's a problem for another day.
            cursor.execute("SELECT * FROM unit_removable_skill_m WHERE effect_range < 999")
            return cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error at '{db_path}': {e}")
        return []

def _map_attribute(record: sqlite3.Row) -> str:
    """Determines the skill's attribute based on its effect and target types."""
    attribute_map = {1: "Smile", 2: "Pure", 3: "Cool"}
    return attribute_map.get(record['effect_type']) or attribute_map.get(record['target_type'], "")

def _map_group(record: sqlite3.Row) -> str:
    """Determines the skill's group (μ's, Aqours, etc.) from its trigger type."""
    group_map = {4: "μ's", 5: "Aqours", 60: "Nijigasaki", 143: "Liella!"}
    return group_map.get(record['trigger_type'], "")

def _map_equip_restriction(record: sqlite3.Row, character_map: Dict[str, str]) -> str:
    """Determines the equipment restriction (year, character, or attribute)."""
    ref_type = record['target_reference_type']
    target_type = record['target_type']

    if ref_type == 1:  # School Year
        return {1: "1st years", 2: "2nd years", 3: "3rd years"}.get(target_type, "")
    elif ref_type == 2:  # Character
        return character_map.get(str(target_type), f"Unknown Character (ID: {target_type})")
    elif ref_type == 3:  # Attribute
        return {1: "Smile", 2: "Pure", 3: "Cool"}.get(target_type, "")
    return ""

def _map_effect(record: sqlite3.Row) -> str:
    """Determines the skill's effect type based on multiple record properties."""
    if record['effect_range'] == 2:
        return "all percent boost"
    elif record['target_reference_type'] == 0:
        return "self flat boost"
    elif record['effect_type'] == 11:
        return "charm"
    elif record['effect_type'] == 12:
        return "heal"
    elif record['effect_type'] in [13, 14, 15]:
        return "trick"
    else:
        return "self percent boost"

def transform_skill_record(record: sqlite3.Row, character_map: Dict[str, str]) -> Dict[str, Any]:
    """Transforms a single database record into JSON object structure."""
    value = record['effect_value']
    if record['fixed_value_flag'] != 1:
        value = round(value / 100.0, 3)

    return {
        "id": record['unit_removable_skill_id'],
        "name": record['name_en'],
        "effect": _map_effect(record),
        "slots": record['size'],
        "attribute": _map_attribute(record),
        "group": _map_group(record),
        "equip_restriction": _map_equip_restriction(record, character_map),
        "target": "self" if record['effect_range'] == 1 else "all",
        "value": value
    }

def write_json_file(data: List[Dict[str, Any]], path: str) -> None:
    """Writes the provided data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully created '{path}' with {len(data)} entries.")
    except IOError as e:
        print(f"Error writing to file '{path}': {e}")

def generate_sis_data():
    """Main function to orchestrate the data processing pipeline."""
    character_map = load_character_map(CHARACTERS_JSON_PATH)
    if not character_map:
        print("Character map could not be loaded. Exiting.")
        return

    skill_records = fetch_skill_records(DATABASE_PATH)
    if not skill_records:
        print("No skill records found or database could not be read. Exiting.")
        return

    processed_skills = [transform_skill_record(record, character_map) for record in skill_records]

    write_json_file(processed_skills, OUTPUT_JSON_PATH)

if __name__ == '__main__':
    generate_sis_data()
