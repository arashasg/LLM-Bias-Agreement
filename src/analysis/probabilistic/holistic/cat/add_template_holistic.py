import os
import json
import pandas as pd

def add_templates_to_predictions_optimized(csv_path, predictions_dir):
    """
    Reads a CSV file containing sentence templates, groups them by axis for optimization,
    and updates JSON prediction files by adding the corresponding template.
    
    Robustness: Handles NaN/Float values in both CSV and JSON data.
    """
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory not found at {predictions_dir}")
        return

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        
        # CLEANING STEP:
        # 1. Fill NaN values with empty strings to avoid 'float' errors
        # 2. Force conversion to string type
        # 3. Strip whitespace
        for col in ['text', 'axis', 'template']:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create a nested dictionary for fast lookup: { axis_name: { sentence: template } }
    lookup_map = {}
    try:
        for axis, group in df.groupby('axis'):
            # Convert the group into a dictionary {sentence: template}
            # We explicitly drop empty keys if any slipped through
            valid_rows = group[group['text'] != ""]
            lookup_map[axis] = pd.Series(
                valid_rows.template.values, index=valid_rows.text
            ).to_dict()
    except Exception as e:
        print(f"Error building lookup map: {e}")
        return

    # Iterate over all JSON files in the predictions directory
    for filename in os.listdir(predictions_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(predictions_dir, filename)
            
            try:
                # Read the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                updated_data = []
                match_count = 0
                
                # Process each sample in the JSON file
                for sample in data:
                    # ROBUSTNESS FIX: Force string conversion before stripping
                    # This handles cases where sentence/axis might be None or NaN (parsed as float)
                    raw_sentence = sample.get("sentence", "")
                    raw_axis = sample.get("axis", "")
                    
                    sentence = str(raw_sentence).strip() if raw_sentence is not None else ""
                    axis = str(raw_axis).strip() if raw_axis is not None else ""
                    
                    found_template = None

                    # OPTIMIZATION: First check if we have templates for this axis
                    if axis in lookup_map:
                        # Then check if the sentence exists within that specific axis group
                        if sentence in lookup_map[axis]:
                            found_template = lookup_map[axis][sentence]
                            match_count += 1
                    
                    sample["template"] = found_template
                    updated_data.append(sample)
                
                # Write the updated data back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=4)
                
                print(f"Processed {filename}: Updated {match_count}/{len(data)} samples.")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# Example Usage:
# Replace these paths with your actual file paths
csv_file_path = "data/holistic-bias/sentences.csv"
predictions_folder_path = "predictions/holistic"

# Uncomment the line below to run the function
add_templates_to_predictions_optimized(csv_file_path, predictions_folder_path)