import json
import os
import glob

def combine_json_files(input_pattern, output_file):
    combined_data = []

    # Use glob to find all matching JSON files
    for filename in glob.glob(input_pattern):
        with open(filename, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)  # Merge the data

    # Write the combined data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

# Base directory where the output files are stored
base = "./data/winobiasp/data/"

# Specify the patterns for female and male JSON files
female_input_pattern = os.path.join(base, "output_female_*.json")
male_input_pattern = os.path.join(base, "output_male_*.json")
input_pattern = os.path.join(base, "output_with_occupation_*.json")
# Combine the files
combine_json_files(female_input_pattern, os.path.join(base, "combined_output_female.json"))
combine_json_files(male_input_pattern, os.path.join(base, "combined_output_male.json"))
combine_json_files(input_pattern, os.path.join(base, "combined_output.json"))
print("Combining completed.")