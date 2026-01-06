import json
import os

def generate_model_rankings(input_dir, output_file):
    """
    Reads model analysis files, ranks them based on how close their 
    overall_ratio is to 0.5, and saves the ranking.
    """
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    model_scores = []

    # 1. Iterate and Extract
    for filename in os.listdir(input_dir):
        if filename.endswith("_cat_scores.json"):
            file_path = os.path.join(input_dir, filename)
            
            try:
                # Extract model name from filename
                # Example: "Llama-4-Scout-17B-16E_cat_scores.json" -> "Llama-4-Scout-17B-16E"
                model_name = filename.replace("_cat_scores.json", "")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                overall_ratio = data.get("overall_ratio")
                
                if overall_ratio is not None:
                    # Calculate distance from 0.5
                    distance = abs(overall_ratio - 0.5)
                    
                    model_scores.append({
                        "model_name": model_name,
                        "overall_ratio": overall_ratio,
                        "distance_from_0.5": distance
                    })
                else:
                    print(f"Warning: 'overall_ratio' missing in {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # 2. Sort by distance (ascending - smallest distance is best)
    model_scores.sort(key=lambda x: x["distance_from_0.5"])

    # 3. Assign Ranks
    ranked_results = []
    for rank, score_data in enumerate(model_scores, 1):
        score_data["rank"] = rank
        ranked_results.append(score_data)

    # 4. Save Output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ranked_results, f, indent=4)
        print(f"Successfully saved model rankings to: {output_file}")
        
        # Optional: Print top 3 for verification
        print("\nTop 3 Ranked Models:")
        for item in ranked_results[:3]:
            print(f"{item['rank']}. {item['model_name']} (Distance: {item['distance_from_0.5']:.6f})")
            
    except Exception as e:
        print(f"Error saving output file: {e}")

# --- Configuration ---
if __name__ == "__main__":
    # Update these paths to match your actual directory structure
    INPUT_FOLDER = "output_updated/holistic/cat/per_model_scores"
    OUTPUT_FILE = "output_updated/holistic/cat/model_ranking_cat.json"
    
    generate_model_rankings(INPUT_FOLDER, OUTPUT_FILE)