import os
import json
import numpy as np
from scipy import stats

def get_perplexity_score(item, type_prefix):
    """
    Helper function to retrieve perplexity score using multiple possible keys.
    type_prefix: 'original' or 'replaced'
    """
    # Define possible keys for original and replaced sentences
    if type_prefix == 'original':
        keys = ["sentence_perplexity_score", "original_sentence_perplexity_score"]
    elif type_prefix == 'replaced':
        keys = ["replaced_sentence_perplexity_score"]
    
    for key in keys:
        if key in item:
            return item[key]
    return None

def calculate_group_statistics(data, group_name):
    """Calculate mean and standard deviation for original and replaced perplexity scores of a group."""
    
    original_scores = []
    replaced_scores = []

    for item in data:
        if item.get("group") != group_name:
            continue
            
        orig_score = get_perplexity_score(item, 'original')
        rep_score = get_perplexity_score(item, 'replaced')

        if orig_score is not None and rep_score is not None:
            original_scores.append(orig_score)
            replaced_scores.append(rep_score)

    if len(original_scores) < 2 or len(replaced_scores) < 2:
        return None  # Skip groups with insufficient data for t-test

    stats_dict = {
        "original_mean": np.mean(original_scores),
        "original_std": np.std(original_scores, ddof=1),
        "original_n": len(original_scores),
        "replaced_mean": np.mean(replaced_scores),
        "replaced_std": np.std(replaced_scores, ddof=1),
        "replaced_n": len(replaced_scores),
    }
    return stats_dict

def perform_ttest(mean1, mean2, std1, std2, n1, n2, alpha=0.05):
    """Perform an independent t-test using precomputed statistics."""
    t_stat, p_value = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=n1,
                                                 mean2=mean2, std2=std2, nobs2=n2,
                                                 equal_var=False)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": bool(p_value < alpha)  # Convert NumPy bool to Python bool
    }

def process_json_files(input_dir, output_dir):
    """Process each JSON file, detect unique groups, and perform t-tests for each group."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r") as f:
                data = json.load(f)

            # --- Missing Key Check & Data Validation ---
            missing_count = 0
            valid_data = []
            
            for item in data:
                orig_score = get_perplexity_score(item, 'original')
                rep_score = get_perplexity_score(item, 'replaced')

                if orig_score is None or rep_score is None:
                    missing_count += 1
                else:
                    valid_data.append(item)
            
            if missing_count > 0:
                print(f"File: {filename} - Skipped {missing_count} samples due to missing perplexity keys.")

            # Use only valid data for further processing
            # Detect unique groups in the valid data
            unique_groups = set(item["group"] for item in valid_data if "group" in item)

            group_results = {}
            for group in unique_groups:
                stats_dict = calculate_group_statistics(valid_data, group)
                if stats_dict:  # Ensure there is enough data to compute statistics
                    result = perform_ttest(
                        stats_dict["original_mean"], stats_dict["replaced_mean"],
                        stats_dict["original_std"], stats_dict["replaced_std"],
                        stats_dict["original_n"], stats_dict["replaced_n"]
                    )
                    group_results[group] = {"statistics": stats_dict, "t_test_results": result}

            with open(output_path, "w") as f:
                json.dump(group_results, f, indent=4)

            print(f"Processed: {filename} -> {output_path}")

# Example usage
input_directory = "predictions/reddit"
output_directory = "output_updated/reddit/lmb/per_model_scores"
process_json_files(input_directory, output_directory)