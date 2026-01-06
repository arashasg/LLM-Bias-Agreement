import json
import os
import glob

def rank_models_by_perplexity_difference(input_folder, output_file_path):
    """
    Ranks models based on the difference between the mean perplexity 
    of female and male sentences.
    
    Args:
        input_folder (str): Path to the folder containing the JSON data files.
        output_file_path (str): Full path (including filename) where the output JSON will be saved.
    """
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_stats = []

    # Get all JSON files in the directory ending with _perplexity_scores.json
    search_pattern = os.path.join(input_folder, "*_perplexity_scores.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching '*_perplexity_scores.json' found in {input_folder}")
        return

    print(f"Processing {len(files)} files...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract model name based on the specific suffix
        model_name = filename.replace("_perplexity_scores.json", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract statistics directly from the root structure
            stats = data.get('statistics', {})
            
            if not stats:
                print(f"Warning: No stats found in {filename}")
                continue
            
            # Get means
            female_mean = stats.get('female_mean', 0.0)
            male_mean = stats.get('male_mean', 0.0)
            
            # Calculate Difference (absolute value)
            # Smaller difference implies less gender bias in perplexity scores
            diff = abs(female_mean - male_mean)
            
            model_stats.append({
                'model': model_name,
                'female_mean': female_mean,
                'male_mean': male_mean,
                'difference': diff
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Rank Models
    # Sort by difference ascending (smaller difference = better performance/less bias)
    ranked_models = sorted(model_stats, key=lambda x: x['difference'])

    # Add Rank to the dictionary
    for rank, m in enumerate(ranked_models, 1):
        m['rank'] = rank

    # Print Rankings Table to Console
    print("-" * 110)
    print(f"{'Rank':<5} | {'Model Name':<40} | {'Diff':<12} | {'Female Mean':<12} | {'Male Mean':<12}")
    print("-" * 110)
    
    for stats in ranked_models:
        print(f"{stats['rank']:<5} | {stats['model']:<40} | {stats['difference']:.4f}       | {stats['female_mean']:.4f}       | {stats['male_mean']:.4f}")
    print("-" * 110)

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
    input_directory = "output_updated/bug/lmb/per_model_scores" 
    output_json_path = "output_updated/bug/lmb/model_rankings_lmb.json"
    
    rank_models_by_perplexity_difference(input_directory, output_json_path)