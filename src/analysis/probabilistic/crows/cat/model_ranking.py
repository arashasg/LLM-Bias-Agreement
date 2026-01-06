import json
import os
import glob

def generate_model_rankings_from_cat_scores(input_folder, output_file_path):
    """
    Ranks models based on the distance of their 'overall' ratio from 0.5.
    
    Args:
        input_folder (str): Path to the folder containing the JSON data files.
        output_file_path (str): Full path where the output JSON will be saved.
    """
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_scores = []

    # Get all JSON files in the directory matching the pattern
    search_pattern = os.path.join(input_folder, "*_cat_scores.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching '*_cat_scores.json' found in {input_folder}")
        return

    print(f"Processing {len(files)} files...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract model name: remove '_cat_scores.json'
        model_name = filename.replace("_cat_scores.json", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Overall Ratio
            overall_data = data.get("overall", {})
            ratio = overall_data.get("ratio")
            
            if ratio is not None:
                # Calculate distance from 0.5 (smaller is better)
                distance = abs(ratio - 0.5)
                
                model_scores.append({
                    "model_name": model_name,
                    "ratio": ratio,
                    "distance_from_0.5": distance
                })
            else:
                print(f"Warning: 'overall.ratio' missing in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Rank Models
    # Sort by distance ascending (smaller distance = rank 1)
    ranked_models = sorted(model_scores, key=lambda x: x['distance_from_0.5'])

    # Add Rank to the dictionary
    for rank, m in enumerate(ranked_models, 1):
        m['rank'] = rank

    # Print Rankings Table to Console (Optional)
    print("-" * 80)
    print(f"{'Rank':<5} | {'Model Name':<35} | {'Ratio':<10} | {'Dist from 0.5':<15}")
    print("-" * 80)
    
    for stats in ranked_models:
        print(f"{stats['rank']:<5} | {stats['model_name']:<35} | {stats['ratio']:.4f}     | {stats['distance_from_0.5']:.4f}")
    print("-" * 80)

    # Save to JSON file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(ranked_models, f, indent=4)
        print(f"\nRankings successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

# Example Usage
if __name__ == "__main__":
    # Change these paths to your actual directories/files
    input_directory = "output_updated/crows/cat/per_model_cat_scores" 
    output_json_path = "output_updated/crows/cat/model_ranking.json"
    
    generate_model_rankings_from_cat_scores(input_directory, output_json_path)