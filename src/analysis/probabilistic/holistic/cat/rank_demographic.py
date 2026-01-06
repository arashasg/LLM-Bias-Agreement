import json
import os
import pandas as pd

def generate_demographic_rankings_csv(input_folder_path, output_csv_path):
    """
    Reads JSON files from the input folder, calculates demographic axis rankings 
    based on the deviation of the CAT score ratio from the gold standard (0.5), 
    and saves a consolidated CSV report.
    
    Structure:
    - Rows: Demographic Axes
    - Columns: Models + Mean Rank + Final Rank
    """
    
    if not os.path.exists(input_folder_path):
        print(f"Error: Input folder '{input_folder_path}' does not exist.")
        return

    # Dictionary to store rankings: {ModelName: {Axis: Rank}}
    model_axis_rankings = {}

    # 1. Iterate through files in the folder
    files = [f for f in os.listdir(input_folder_path) if f.endswith('_scores.json')]
    
    if not files:
        print(f"No '_info.json' files found in {input_folder_path}")
        return

    print(f"Processing {len(files)} files...")

    for filename in files:
        # Extract model name: remove '_info.json'
        model_name = filename.replace('_info.json', '')
        file_path = os.path.join(input_folder_path, filename)

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            axis_diffs = []
            
            # Access the 'axis_breakdown' dictionary where the metrics are stored
            axis_breakdown = data.get("axis_breakdown", {})

            # 2. Calculate differences for this model
            for axis, content in axis_breakdown.items():
                ratio = content.get("ratio")

                if ratio is not None:
                    # Metric: Absolute difference between ratio and 0.5 (gold standard)
                    # [cite_start]Smaller difference means closer to 0.5, which is better/less biased [cite: 139]
                    diff = abs(ratio - 0.5)
                    axis_diffs.append({'axis': axis, 'diff': diff})
                else:
                    print(f"Warning: Missing 'ratio' for axis '{axis}' in model '{model_name}'.")

            # 3. Rank axes for this model (1 = smallest difference/best)
            # Sort by difference ascending
            axis_diffs.sort(key=lambda x: x['diff'])
            
            # Store ranks in a dictionary {Axis: Rank}
            ranks = {item['axis']: i + 1 for i, item in enumerate(axis_diffs)}
            model_axis_rankings[model_name] = ranks

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # 4. Create DataFrame (Rows=Axes, Cols=Models)
    # We load the dict of dicts into a DataFrame
    df = pd.DataFrame(model_axis_rankings)
    
    # 5. Calculate Mean Ranking for each axis
    # axis=1 averages across columns (models). Pandas ignores NaNs by default.
    df['Mean Rank'] = df.mean(axis=1)

    # 6. Calculate Final Rank based on Mean Ranking
    # method='min' handles ties by assigning the lowest rank to the group
    # Ascending=True because lower Mean Rank is better
    df['Final Rank'] = df['Mean Rank'].rank(method='min', ascending=True).astype(int)

    # Sort by Final Rank for better readability
    df = df.sort_values(by='Final Rank')

    # 7. Save to CSV
    try:
        df.to_csv(output_csv_path, index_label="Demographic Axis")
        print(f"Successfully saved rankings to {output_csv_path}")
        print("\nPreview of Results:")
        print(df[['Mean Rank', 'Final Rank']].head())
    except Exception as e:
        print(f"Error saving CSV: {e}")

# Example Usage:
# Replace these paths with your actual folder paths
input_folder = "output_updated/holistic/cat/per_model_scores" 
output_file = "output_updated/holistic/cat/demographic_axis_rankings.csv"

if __name__ == "__main__":
    # Ensure this block only runs if paths are valid/set
    if os.path.exists(input_folder):
        generate_demographic_rankings_csv(input_folder, output_file)
    else:
        print("Please update the 'input_folder' variable with a valid path.")