import json
import csv
import os
import re
from collections import OrderedDict

# --- Configuration ---
DATA_DIR = "./data"
INPUT_JSON_PATH = os.path.join(DATA_DIR, "accessories_old.json")
INPUT_CSV_PATH = os.path.join(DATA_DIR, "Copy of Encore stuff - AccData.csv")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "accessories.json")

# --- Constants ---
# These are lower rarity accessories that the original accessory data did not contain.
# They will be manually added.
IDS_TO_CREATE = {
    '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62',
    '63', '129', '130', '131', '174', '175', '176', '179', '180', '181', '219',
    '220', '221'
}

class _NoIndent:
    """A wrapper to signal that a list should not be indented."""
    def __init__(self, value):
        self.value = value

class _CompactJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that prepares lists for single-line formatting
    by wrapping them with a placeholder.
    """
    _placeholder = "@@@COMPACT_LIST@@@"

    def default(self, o):
        """Overrides the default JSONEncoder method."""
        if isinstance(o, _NoIndent):
            return f"{self._placeholder}{json.dumps(o.value)}{self._placeholder}"
        return super().default(o)

def _safe_cast(value, cast_type, default=0):
    """Safely casts a value to a given type, returning a default on failure."""
    if value is None or value == '':
        return default
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default

def _build_list_with_fill_forward(row, key_prefix, cast_type, default_value=0):
    """Builds a list of 16 values, carrying forward the last valid value for blanks."""
    result_list = []
    last_value = default_value
    for i in range(16):
        raw_value = row.get(f'{key_prefix}{i}')
        if raw_value is not None and raw_value.strip() != '':
            last_value = _safe_cast(raw_value, cast_type, default=last_value)
        result_list.append(last_value)
    return result_list

def _build_stats_list_with_fill_forward(row, default_value=None):
    """Builds the stats list, carrying forward the last valid value for blanks."""
    if default_value is None:
        default_value = [0, 0, 0]

    result_list = []
    last_value = default_value
    for i in range(16):
        smile = row.get(f'accessories/smile/{i}')
        pure = row.get(f'accessories/pure/{i}')
        cool = row.get(f'accessories/cool/{i}')

        if all(v is not None and v.strip() != '' for v in [smile, pure, cool]):
            last_value = [
                _safe_cast(smile, int, default=last_value[0]),
                _safe_cast(pure, int, default=last_value[1]),
                _safe_cast(cool, int, default=last_value[2])
            ]
        result_list.append(last_value)
    return result_list

def load_json_data(filepath):
    """Loads and returns data from a JSON file."""
    print(f"Loading JSON data from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'. Ensure it's valid.")
        return None

def load_csv_data(filepath):
    """Loads and returns data from a CSV file as a list of dictionaries."""
    print(f"Loading CSV data from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except (IOError, csv.Error) as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

def _create_new_accessory_entry(row):
    """Creates a new accessory entry from a CSV row."""
    new_entry = OrderedDict()
    new_entry['character'] = row.get('accessories/effect_target_name')
    new_entry['name'] = row.get('accessories/name_en')
    new_entry['card_id'] = row.get('accessories/unit_stub/ordinal')

    new_entry['stats'] = _build_stats_list_with_fill_forward(row)

    new_entry['skill'] = OrderedDict([
        ('trigger', OrderedDict([
            ('chances', _build_list_with_fill_forward(row, 'accessories/probability/', int)),
            ('values', _build_list_with_fill_forward(row, 'accessories/trigger_val/', int))
        ])),
        ('effect', OrderedDict([
            ('type', row.get('Skill')),
            ('durations', _build_list_with_fill_forward(row, 'accessories/effect_dur/', float)),
            ('values', _build_list_with_fill_forward(row, 'accessories/effect_val/', int))
        ]))
    ])
    return new_entry

def process_and_update_data(json_data, csv_data):
    """
    Merges, modifies, and adds entries to the JSON data based on the CSV data.
    """
    print("Processing and updating data...")
    modified_count, added_count = 0, 0
    unmatched_ids = []
    json_keys = set(json_data.keys())

    for row in csv_data:
        accessory_id = row.get('accessories/id')
        if accessory_id in json_keys:
            modified_count += 1
            original_entry = json_data[accessory_id]

            updated_entry = OrderedDict()
            updated_entry['character'] = row.get('accessories/unit_stub/char_name', original_entry.get('character'))
            updated_entry['name'] = row.get('accessories/name_en')
            updated_entry['card_id'] = row.get('accessories/unit_stub/ordinal')
            updated_entry['stats'] = original_entry.get('stats')

            skill_data = original_entry.get('skill', {}).copy()
            if skill_data and 'effect' in skill_data:
                skill_data['effect']['type'] = row.get('Skill')

            updated_entry['skill'] = skill_data
            json_data[accessory_id] = updated_entry
        elif accessory_id in IDS_TO_CREATE:
            added_count += 1
            json_data[accessory_id] = _create_new_accessory_entry(row)
        else:
            unmatched_ids.append(accessory_id)

    print(f"Processed {len(csv_data)} rows from CSV. Modified {modified_count} entries and added {added_count} new entries.")

    if unmatched_ids:
        print("\nWarning: The following accessory IDs from the CSV were not found in the JSON and were skipped:")
        print(", ".join(filter(None, unmatched_ids)))

    return json_data

def save_modified_json(data, filepath):
    """Saves the given data to a file using custom JSON formatting."""
    print(f"Saving modified data to '{filepath}'...")

    for item in data.values():
        if isinstance(item, dict):
            if 'stats' in item and isinstance(item['stats'], list):
                item['stats'] = _NoIndent(item['stats'])
            if 'skill' in item and isinstance(item.get('skill'), dict):
                for category in ['trigger', 'effect']:
                    if category in item['skill'] and isinstance(item['skill'][category], dict):
                        for key, value in item['skill'][category].items():
                            if isinstance(value, list):
                                item['skill'][category][key] = _NoIndent(value)

    try:
        json_string_with_placeholders = json.dumps(data, cls=_CompactJSONEncoder, indent=2)

        placeholder_regex = r'"@@@COMPACT_LIST@@@(.*?)@@@COMPACT_LIST@@@"'
        final_json_string = re.sub(placeholder_regex, r'\1', json_string_with_placeholders)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_json_string)
        print("Successfully saved the file.")
    except IOError as e:
        print(f"Error: Could not write to the file '{filepath}'. Reason: {e}")

def main():
    accessory_data = load_json_data(INPUT_JSON_PATH)
    csv_rows = load_csv_data(INPUT_CSV_PATH)

    if accessory_data and csv_rows:
        updated_data = process_and_update_data(accessory_data, csv_rows)
        sorted_data = OrderedDict(sorted(updated_data.items(), key=lambda item: int(item[0])))
        save_modified_json(sorted_data, OUTPUT_JSON_PATH)

    print("Script finished.")

if __name__ == "__main__":
    main()
