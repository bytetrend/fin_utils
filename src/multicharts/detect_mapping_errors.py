#!/usr/bin/env python3
"""
Script to verify that mapping file entries match between mc_symbol:mc_name
and broker_symbol:br_name for all the mapping entries.
"""

import json

def load_mapping_file(file_path):
    """Load the mapping file and return the data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return None

def verify_mapping_entries(mapping_data):
    """Verify that mc_symbol:mc_name matches broker_symbol:br_name for all entries."""
    errors = []

    # Handle the case where mapping_data is a dictionary with "mapping" key
    if isinstance(mapping_data, dict) and 'mapping' in mapping_data:
        entries = mapping_data['mapping']
    else:
        entries = mapping_data

    if not isinstance(entries, list):
        print("Error: Mapping data should be a list of entries.")
        return errors

    for i, entry in enumerate(entries):
        # Check if entry has the required fields
        if 'mc_symbol' not in entry or 'broker_symbol' not in entry:
            errors.append(f"Entry {i} is missing required fields")
            continue

        # Extract mc_name from mc_symbol object
        mc_symbol = entry['mc_symbol']
        broker_symbol = entry['broker_symbol']

        # Check if both have the required name fields
        if 'mc_name' not in mc_symbol or 'br_name' not in broker_symbol:
            errors.append(f"Entry {i} is missing mc_name or br_name fields")
            continue

        mc_name = mc_symbol['mc_name']
        br_name = broker_symbol['br_name']

        # Check if mc_name matches br_name
        if mc_name != br_name:
            errors.append(f"Entry {i}: mc_name '{mc_name}' does not match br_name '{br_name}'")

    return errors

def main():
    """Main function to run the verification."""
    file_path = 'src/multicharts/IB_mapping.json'

    print("Verifying mapping file...")
    mapping_data = load_mapping_file(file_path)

    if mapping_data is None:
        return
    errors = verify_mapping_entries(mapping_data)

    if errors:
        print("\nFound the following mismatches:")
        for error in errors:
            print(f"  - {error}")
        print(f"\nTotal mismatches found: {len(errors)}")
    else:
        print("\nAll mapping entries match correctly!")
        print("mc_symbol:mc_name == broker_symbol:br_name for all entries.")

if __name__ == "__main__":
    main()