import json
import os
import glob

def generate_demographic_rankings(input_folder, output_file):
    """
    Generates rankings for demographic axes based on statistical significance and mean perplexity difference.
    
    Args:
        input_folder (str): Path to the folder containing input JSON files.
        output_file (str): Path where the output JSON file will be saved.
    """
    
    all_model_rankings = {}
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Iterate over files in the directory
    # Only processes files ending with _perplexity_scores.json
    files = glob.glob(os.path.join(input_folder, "*_perplexity_scores.json"))
    
    if not files:
        print(f"No matching files found in {input_folder}")
        return

    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract model name by removing the suffix
        model_name = filename.replace("_perplexity_scores.json", "")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Prepare list for sorting
            axis_data = []
            
            for axis, details in data.items():
                stats = details.get('statistics', {})
                ttest = details.get('t_test_results', {})
                
                # Get significance (True/False)
                is_significant = ttest.get('is_significant', False)
                
                # Get means to calculate difference
                original_mean = stats.get('original_mean', 0)
                replaced_mean = stats.get('replaced_mean', 0)
                
                # Calculate absolute difference between means
                mean_diff = abs(original_mean - replaced_mean)
                
                axis_data.append({
                    'axis': axis,
                    'is_significant': is_significant,
                    'mean_diff': mean_diff
                })
            
            # Sort the axes
            # Primary key: is_significant (False < True). 
            #   False (Not Significant) -> Lower Rank (Better)
            #   True (Significant) -> Higher Rank (Worse)
            # Secondary key: mean_diff (Ascending). 
            #   Lower difference -> Lower Rank (Better)
            sorted_axes = sorted(axis_data, key=lambda x: (x['is_significant'], x['mean_diff']))
            
            # Assign ranks and create result list for this model
            ranked_output = []
            for rank, item in enumerate(sorted_axes, 1):
                ranked_output.append({
                    'rank': rank,
                    'axis': item['axis'],
                    'is_significant': item['is_significant'],
                    'mean_diff': item['mean_diff']
                })
            
            all_model_rankings[model_name] = ranked_output
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

    # Write result to output file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_model_rankings, f, indent=4)
        print(f"Successfully generated rankings in '{output_file}'")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

# Example usage
# Replace 'input_folder_path' and 'output_file.json' with your actual paths
if __name__ == "__main__":
    generate_demographic_rankings('output_updated/reddit/lmb/per_model_scores', 'output_updated/reddit/lmb/demographic_rankings.json')