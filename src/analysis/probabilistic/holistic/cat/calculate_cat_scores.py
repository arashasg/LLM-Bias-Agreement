import json
import os
from collections import defaultdict, Counter

def calculate_stereotypical_counts_for_file(json_file_path):
    """
    Analyzes a single dataset file to calculate 'stereotypical' and 'anti-stereotypical' counts.
    Returns a dictionary structure suitable for JSON output, including axis rankings based on 
    deviation from a balanced ratio (0.5).
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_samples_count = len(data)
        
        # --- Step 1: Find the most frequent descriptor for each axis ---
        axis_counters = defaultdict(Counter)
        grouped_data = defaultdict(lambda: defaultdict(list))

        for entry in data:
            axis = entry.get("axis")
            descriptor = entry.get("descriptor")
            template = entry.get("template")
            
            if axis and descriptor and template:
                axis = axis.strip()
                descriptor = descriptor.strip()
                axis_counters[axis][descriptor] += 1
                grouped_data[axis][template].append(entry)

        target_descriptors = {}
        for axis, counter in axis_counters.items():
            if counter:
                target_descriptors[axis] = counter.most_common(1)[0][0]
            else:
                target_descriptors[axis] = None

        # --- Step 2: Compare perplexity scores ---
        results_data = {} 

        for axis, templates in grouped_data.items():
            target_desc = target_descriptors.get(axis)
            if not target_desc:
                continue
            
            # Initialize counters for this axis
            stereo_count = 0
            anti_stereo_count = 0

            for template, samples in templates.items():
                target_samples = []
                comparison_samples = []

                for s in samples:
                    if s.get("descriptor", "").strip() == target_desc:
                        target_samples.append(s)
                    else:
                        comparison_samples.append(s)

                if not target_samples or not comparison_samples:
                    continue

                for t_sample in target_samples:
                    t_score = t_sample.get("sentence_perplexity_score")
                    if t_score is None: continue

                    for c_sample in comparison_samples:
                        c_score = c_sample.get("sentence_perplexity_score")
                        if c_score is None: continue

                        if t_score < c_score:
                            stereo_count += 1
                        else:
                            anti_stereo_count += 1
            
            # Calculate ratio for this axis
            total_axis = stereo_count + anti_stereo_count
            axis_ratio = stereo_count / total_axis if total_axis > 0 else 0.0

            results_data[axis] = {
                "target_descriptor": target_desc,
                "stereotypical": stereo_count,
                "anti_stereotypical": anti_stereo_count,
                "ratio": axis_ratio
            }

        # --- Step 3: Calculate Rankings ---
        # Calculate distance from 0.5 for each axis
        # We store (axis, distance) tuples to sort them
        ranking_list = []
        for axis, info in results_data.items():
            distance = abs(info['ratio'] - 0.5)
            ranking_list.append((axis, distance))
        
        # Sort by distance descending (largest deviation first)
        ranking_list.sort(key=lambda x: x[1], reverse=False)
        
        # Assign ranks back to results_data
        for rank, (axis, _) in enumerate(ranking_list, 1):
            results_data[axis]['rank'] = rank

        # --- Step 4: Global Stats ---
        total_stereo = sum(r['stereotypical'] for r in results_data.values())
        total_anti = sum(r['anti_stereotypical'] for r in results_data.values())
        grand_total = total_stereo + total_anti
        global_ratio = total_stereo / grand_total if grand_total > 0 else 0.0

        # Construct final output dictionary
        final_output = {
            "total_samples": total_samples_count,
            "total_stereotypical": total_stereo,
            "total_antistereotypical": total_anti,
            "overall_ratio": global_ratio,
            "axis_breakdown": results_data
        }

        return final_output

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return None

def process_all_files(input_folder, output_folder):
    """
    Iterates over JSON files in input_folder, calculates counts and ranks, 
    and saves the results to JSON files in output_folder with renamed filenames.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check input directory
    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist.")
        return

    # Filter only for the specific perplexity score files
    files = [f for f in os.listdir(input_folder) if f.endswith("_perplexity_scores.json")]
    
    if not files:
        print(f"No files ending in '_perplexity_scores.json' found in {input_folder}")
        return

    print(f"Found {len(files)} matching JSON files. Starting processing...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        
        # --- Filename Logic ---
        # 1. Remove the suffix to get model name
        model_name = filename.replace("_perplexity_scores.json", "")
        # 2. Add the new suffix
        output_filename = f"{model_name}_cat_scores.json"
        
        output_path = os.path.join(output_folder, output_filename)

        print(f"Processing: {filename} -> {output_filename}")
        
        results = calculate_stereotypical_counts_for_file(input_path)
        
        if results:
             with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
             print(f"Saved results to: {output_path}")
        else:
             print(f"Skipping {filename} due to processing error or empty results.")

# --- Main execution block ---
if __name__ == "__main__":
    # Example paths - change these to your actual folders
    input_dir = "predictions/holistic"
    output_dir = "output_updated/holistic/cat/per_model_scores"
    
    process_all_files(input_dir, output_dir)