import os
import csv
import json
import re

def rank_models_by_emt(input_folder, output_file_path):
    """
    Reads CSV files containing toxicity metrics, extracts the EMT score,
    ranks models (lower EMT is better), and saves the ranking to a JSON file.
    """
    
    # Dictionary to ensure unique model names: { "model_name": emt_score }
    # If a duplicate is found (unlikely with correct filtering), we keep the better (lower) score.
    best_model_scores = {}
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(input_folder):
        # 1. STRICT FILTER: Only process 'overall_metrics' CSV files
        if not filename.endswith(".csv") or "overall_metrics" not in filename:
            continue
            
        file_path = os.path.join(input_folder, filename)
        model_name = None

        # --- LOGIC TO EXTRACT MODEL NAME ---
        
        # Case 1: Long format with parameters 
        # e.g. "llama4_scout_temprature_1.0..._predictions_overall_metrics.csv"
        if "_temprature" in filename:
            model_name = filename.split("_temprature")[0]
            
        # Case 2: Short format 
        # e.g. "Meta-Llama-3-8B_overall_metrics.csv"
        else:
            # Safely remove the suffix
            model_name = filename.replace("_overall_metrics.csv", "")
            
        # -----------------------------------

        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                emt_value = None
                
                # Find the row where Metric is 'EMT'
                for row in reader:
                    if row.get('Metric') == 'EMT':
                        try:
                            emt_value = float(row.get('Mean'))
                        except ValueError:
                            print(f"Error parsing EMT value in {filename}")
                        break
                
                if emt_value is not None:
                    # Deduplication Logic:
                    # If model exists, only overwrite if new score is better (lower)
                    if model_name not in best_model_scores:
                        best_model_scores[model_name] = emt_value
                    else:
                        if emt_value < best_model_scores[model_name]:
                            best_model_scores[model_name] = emt_value
                else:
                    print(f"Warning: EMT metric not found in {filename}")
                    
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Convert dict to list for sorting
    ranked_list = [{'model_name': k, 'emt': v} for k, v in best_model_scores.items()]

    # Rank models: Sort by EMT ascending (Lower EMT = Better Rank)
    ranked_list.sort(key=lambda x: x['emt'])
    
    # Construct the output dictionary
    output_data = {}
    for rank, item in enumerate(ranked_list, 1):
        output_data[item['model_name']] = {
            "toxicity_rank": rank
        }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to JSON
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved rankings for {len(output_data)} models to {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

# Example Usage:
input_dir = "Bold_Generation_Based_Toxcitiy/toxicity_metrics_applied"
output_path = "Bold_Generation_Based_Toxcitiy/toxicity_metrics_applied/model_toxicity_rankings.json"
rank_models_by_emt(input_dir, output_path)