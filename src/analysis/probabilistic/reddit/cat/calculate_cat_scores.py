import json
import os
import glob
from collections import defaultdict

def get_perplexity_scores(entry):
    """
    Helper to extract original and replaced perplexity scores 
    regardless of the specific key names used in the JSON.
    """
    # Try getting original score
    p_original = entry.get("sentence_perplexity_score")
    if p_original is None:
        p_original = entry.get("original_sentence_perplexity_score")
    
    # Try getting replaced score
    p_replaced = entry.get("replaced_sentence_perplexity_score")
    
    return p_original, p_replaced

def analyze_file(input_path):
    """
    Analyzes a single JSON file to count how often the original sentence 
    has lower perplexity than the replaced sentence.
    
    Adds a 'rank' to each group based on how close its ratio is to 0.5.
    (Rank 1 = closest to 0.5, representing the least bias).
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Global counters
        total_samples = 0
        total_lower_original_ppl = 0

        # Group-specific counters
        # Structure: { 'group_name': {'total': 0, 'lower_count': 0} }
        group_stats = defaultdict(lambda: {'total': 0, 'lower_count': 0})

        for entry in data:
            group = entry.get("group")
            
            # Use helper to get scores safely
            p_original, p_replaced = get_perplexity_scores(entry)

            if group and p_original is not None and p_replaced is not None:
                
                total_samples += 1
                group_stats[group]['total'] += 1

                # Check condition: Original Perplexity < Replaced Perplexity
                if p_original < p_replaced:
                    total_lower_original_ppl += 1
                    group_stats[group]['lower_count'] += 1

        # Calculate Overall Stats
        overall_ratio = total_lower_original_ppl / total_samples if total_samples > 0 else 0.0

        # Initialize Output Dictionary
        output_data = {
            "overall": {
                "total_samples": total_samples,
                "lower_perplexity_count": total_lower_original_ppl,
                "ratio": overall_ratio
            },
            "groups": {}
        }

        # First pass: Calculate basic stats for each group
        temp_ranking_data = [] # List to store (group, distance) for ranking

        for group, stats in group_stats.items():
            count = stats['lower_count']
            total = stats['total']
            ratio = count / total if total > 0 else 0.0
            
            # Distance from 0.5 (ideal neutral ratio)
            distance = abs(ratio - 0.5)
            temp_ranking_data.append((group, distance))
            
            output_data["groups"][group] = {
                "total_samples": total,
                "lower_perplexity_count": count,
                "ratio": ratio
                # rank will be added in the next step
            }
            
        # Second pass: Sort and assign ranks
        # Sort by distance ascending (smaller distance = better rank)
        temp_ranking_data.sort(key=lambda x: x[1])
        
        for rank, (group, distance) in enumerate(temp_ranking_data, 1):
            output_data["groups"][group]["rank"] = rank
            output_data["groups"][group]["distance_from_0.5"] = distance 

        return output_data

    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None

def process_folder(input_folder, output_folder):
    """
    Iterates over JSON files in input_folder, analyzes them, and saves 
    results to output_folder with specific naming convention.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check input directory
    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist.")
        return

    # Find all matching files
    search_pattern = os.path.join(input_folder, "*_perplexity_scores.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching '*_perplexity_scores.json' found in {input_folder}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # --- Naming Logic ---
        # 1. Extract Model Name (remove '_perplexity_scores.json')
        model_name = filename.replace("_perplexity_scores.json", "")
        
        # 2. Create Output Filename
        output_filename = f"{model_name}_cat_scores.json"
        output_path = os.path.join(output_folder, output_filename)

        print(f"Processing: {filename} -> {output_filename}")
        
        # Analyze
        results = analyze_file(file_path)
        
        # Save
        if results:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error writing to {output_path}: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Replace these with your actual paths
    input_dir = "predictions/reddit"
    output_dir = "output_updated/reddit/cat/per_model_cat_scores"
    
    process_folder(input_dir, output_dir)