import os
import json

def rank_models_by_honest_score(input_folder, output_file_path):
    """
    Reads JSON files containing HONEST metrics, extracts the 'overall_honest_score',
    ranks models (lower score is better), and saves the ranking.
    """
    
    # Dictionary to ensure unique model names: { "model_name": honest_score }
    best_model_scores = {}
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".json") and filename != "model_honest_rankings.json":
            file_path = os.path.join(input_folder, filename)
            
            # Default model name from filename
            model_name_from_file = filename.replace(".json", "")
            
            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 1. Extract Score (New Key: overall_honest_score)
                    honest_score = data.get("overall_honest_score")
                    
                    # 2. Extract Model Name (Prefer internal name if available)
                    internal_name = data.get("model_name")
                    final_model_name = internal_name if internal_name else model_name_from_file
                    
                    if honest_score is not None:
                        score_val = float(honest_score)
                        
                        # Deduplication Logic: Keep the lowest (best) score
                        if final_model_name not in best_model_scores:
                            best_model_scores[final_model_name] = score_val
                        else:
                            if score_val < best_model_scores[final_model_name]:
                                best_model_scores[final_model_name] = score_val
                                
                    else:
                        print(f"Warning: 'overall_honest_score' not found in {filename}")
                        
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Convert dict to list for sorting
    model_scores = [{'model_name': k, 'score': v} for k, v in best_model_scores.items()]

    # Rank models: Sort by score ascending (Lower score = Better/Lower Rank)
    model_scores.sort(key=lambda x: x['score'])
    
    # Construct the output dictionary
    # Structure: { "ModelName": { "honest_rank": 1 } }
    output_data = {}
    for rank, item in enumerate(model_scores, 1):
        output_data[item['model_name']] = {
            "honest_rank": rank
        }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to JSON
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved HONEST rankings for {len(output_data)} models to {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

# Example Usage:
input_dir = "Bold_Generation_Based_Toxcitiy/honest_result"
output_path = "Bold_Generation_Based_Toxcitiy/honest_result/model_honest_rankings.json"
rank_models_by_honest_score(input_dir, output_path)