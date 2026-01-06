import json
import numpy as np
import os
from collections import defaultdict

# ==========================================
# SETUP: Folder Paths
# ==========================================
INPUT_FOLDER = "predictions/holistic"  # Change this to your actual input folder name
OUTPUT_FOLDER = "holistic_output_info" # Change this to your desired output folder name

def calculate_stats(scores_list):
    """
    Helper function to calculate mean, std, and count given a list of scores.
    """
    if not scores_list:
        return None
    
    n = len(scores_list)
    mean_val = np.mean(scores_list)
    
    # Calculate Standard Deviation (ddof=1 for sample std dev)
    # If there is only 1 sample, std dev is 0
    std_val = np.std(scores_list, ddof=1) if n > 1 else 0.0
    
    return {
        "mean_perplexity": float(mean_val),
        "std_dev": float(std_val),
        "sample_count": int(n)
    }

def process_single_file(input_path, output_path):
    """
    Reads a single JSON file, calculates stats, and writes to output path.
    """
    print(f"Processing: {input_path}")
    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Failed to read {input_path}: {e}")
        return

    # Data Structures for aggregation
    # 1. To store scores per descriptor: grouped_data[axis][descriptor] = [scores]
    # 2. To store scores per axis (overall): axis_overall_data[axis] = [scores]
    grouped_data = defaultdict(lambda: defaultdict(list))
    axis_overall_data = defaultdict(list)

    # --- Step 1: Aggregate Scores ---
    for entry in raw_data:
        axis = entry.get("axis")
        descriptor = entry.get("descriptor")
        score = entry.get("sentence_perplexity_score")

        # Skip entries if key data is missing
        if axis is None or descriptor is None or score is None:
            continue

        # Add to specific descriptor list
        grouped_data[axis][descriptor].append(score)
        
        # Add to general axis list
        axis_overall_data[axis].append(score)

    # --- Step 2: Calculate Statistics ---
    final_output = {}

    for axis, descriptors_dict in grouped_data.items():
        final_output[axis] = {
            "axis_overall_statistics": {},
            "descriptors_statistics": {}
        }

        # 2a. Calculate Overall Axis Stats
        all_axis_scores = axis_overall_data[axis]
        final_output[axis]["axis_overall_statistics"] = calculate_stats(all_axis_scores)

        # 2b. Calculate Stats per Descriptor
        for descriptor, scores in descriptors_dict.items():
            stats = calculate_stats(scores)
            final_output[axis]["descriptors_statistics"][descriptor] = stats

    # --- Step 3: Write to Output ---
    print(f"Writing results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

def process_all_files():
    # Check if input directory exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input directory '{INPUT_FOLDER}' not found.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    # Iterate over files in the input folder
    files_processed = 0
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".json"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            
            # --- Name Extraction Logic ---
            # Remove the specific suffix if it exists to get the clean model name
            suffix = "_perplexity_scores.json"
            if filename.endswith(suffix):
                model_name = filename[:-len(suffix)] # Remove suffix
            else:
                model_name = os.path.splitext(filename)[0] # Fallback: just remove .json

            output_filename = f"{model_name}_info.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            process_single_file(input_path, output_path)
            files_processed += 1
    
    if files_processed == 0:
        print("No .json files found in the input folder.")
    else:
        print(f"Done. Processed {files_processed} files.")

if __name__ == "__main__":
    process_all_files()