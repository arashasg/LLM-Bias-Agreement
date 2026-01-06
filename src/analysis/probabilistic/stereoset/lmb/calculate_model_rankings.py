import json
import os
import glob

def rank_models_by_perplexity_difference(input_folder, output_file_path):
    """
    Ranks models based on the difference between the overall weighted mean perplexity 
    of stereotype and anti-stereotype sentences across all bias types.
    
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
            
            total_stereotype_weighted_sum = 0.0
            total_anti_stereotype_weighted_sum = 0.0
            total_samples = 0
            
            # Aggregate stats across all bias types (groups)
            for bias_type, details in data.items():
                stats = details.get('statistics', {})
                
                if not stats:
                    continue
                
                # Get means and counts from the new data format
                stereo_mean = stats.get('stereotype_mean', 0.0)
                stereo_n = stats.get('stereotype_n', 0)
                
                anti_mean = stats.get('anti_stereotype_mean', 0.0)
                anti_n = stats.get('anti_stereotype_n', 0)
                
                # We typically assume n is the same for paired tests, but use explicit n just in case.
                # Using stereotype_n as the primary weight since usually n_stereo == n_anti
                n = stereo_n
                
                # Calculate weighted sums
                total_stereotype_weighted_sum += (stereo_mean * n)
                total_anti_stereotype_weighted_sum += (anti_mean * n)
                
                total_samples += n
            
            # Calculate Overall Means
            if total_samples > 0:
                overall_stereotype_mean = total_stereotype_weighted_sum / total_samples
                overall_anti_stereotype_mean = total_anti_stereotype_weighted_sum / total_samples
                
                # Calculate Difference (absolute value)
                # Smaller difference implies the model treats stereotypes and anti-stereotypes similarly (less bias)
                diff = abs(overall_stereotype_mean - overall_anti_stereotype_mean)
                
                model_stats.append({
                    'model': model_name,
                    'overall_stereotype_mean': overall_stereotype_mean,
                    'overall_anti_stereotype_mean': overall_anti_stereotype_mean,
                    'difference': diff
                })
            else:
                print(f"Warning: No valid samples found in {filename}")

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
    print(f"{'Rank':<5} | {'Model Name':<40} | {'Diff':<12} | {'Stereo Mean':<12} | {'Anti Mean':<12}")
    print("-" * 110)
    
    for stats in ranked_models:
        print(f"{stats['rank']:<5} | {stats['model']:<40} | {stats['difference']:.4f}       | {stats['overall_stereotype_mean']:.4f}       | {stats['overall_anti_stereotype_mean']:.4f}")
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
    input_directory = "output_updated/stereoset/lmb/per_model_scores" 
    output_json_path = "output_updated/stereoset/lmb/model_rankings_lmb.json"
    
    rank_models_by_perplexity_difference(input_directory, output_json_path)