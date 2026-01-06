import os
import json
import pandas as pd
import glob

def extract_demographic_ranks_to_csv(input_folder_path, output_csv_path):
    """
    Reads JSON files from a folder, extracts demographic rankings, 
    and saves them as a CSV matrix (Models x Demographics).
    """
    
    # List to store the flat data before pivoting
    extracted_data = []

    # Ensure input path exists
    if not os.path.exists(input_folder_path):
        print(f"Error: Input folder not found at {input_folder_path}")
        return

    # iterate through all json files in the folder
    json_files = glob.glob(os.path.join(input_folder_path, "*.json"))
    
    if not json_files:
        print("No JSON files found in the directory.")
        return

    print(f"Found {len(json_files)} files. Processing...")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        # 1. Extract Model Name
        # Logic: Replaces the known suffix with empty string to get model name
        if "_cat_scores.json" in filename:
            model_name = filename.replace("_cat_scores.json", "")
        else:
            # Fallback if file doesn't match expected pattern
            model_name = filename.replace(".json", "")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check if 'groups' key exists
                if "groups" in data:
                    groups_data = data["groups"]
                    
                    # 2. Iterate through demographic groups
                    for group_name, group_info in groups_data.items():
                        # Extract the rank
                        rank = group_info.get("rank")
                        
                        if rank is not None:
                            extracted_data.append({
                                "Model": model_name,
                                "Demographic": group_name,
                                "Rank": rank
                            })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # 3. Create DataFrame
    if not extracted_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(extracted_data)

    # 4. Pivot the data
    # Rows = Model, Columns = Demographic, Values = Rank
    pivot_df = df.pivot(index="Model", columns="Demographic", values="Rank")

    # Optional: Fill NaNs or leave empty if a model misses a demographic
    # pivot_df = pivot_df.fillna("N/A") 

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. Save to CSV
    pivot_df.to_csv(output_csv_path)
    print(f"Successfully saved demographic rankings to: {output_csv_path}")
    print("\nPreview of data:")
    print(pivot_df.head())

# --- Usage ---
if __name__ == "__main__":
    # Define your specific paths here
    INPUT_PATH = "output_updated/reddit/cat/per_model_cat_scores"
    OUTPUT_FILE = "output_updated/reddit/cat/consolidated_demographic_ranks.csv"
    
    # Create dummy folder for testing purposes (Remove this block if running on real data)
    # or simply ensure your INPUT_PATH exists before running.
    
    extract_demographic_ranks_to_csv(INPUT_PATH, OUTPUT_FILE)