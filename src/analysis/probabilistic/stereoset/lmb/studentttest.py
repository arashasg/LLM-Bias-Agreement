import os
import json
import numpy as np
from scipy import stats

def calculate_group_statistics(data, bias_type):
    """Calculate mean and standard deviation for stereotype and anti-stereotype perplexity scores of a bias type."""
    # Using 'bias_type' to filter and specific score keys from the new data format
    stereotype_scores = [
        item["stereotype_perplexity_score"] 
        for item in data 
        if item.get("bias_type") == bias_type and "stereotype_perplexity_score" in item
    ]
    
    anti_stereotype_scores = [
        item["anti_stereotype_perplexity_score"] 
        for item in data 
        if item.get("bias_type") == bias_type and "anti_stereotype_perplexity_score" in item
    ]

    # Ensure we have enough data points for variance calculation (at least 2)
    if len(stereotype_scores) < 2 or len(anti_stereotype_scores) < 2:
        return None

    stats_dict = {
        "stereotype_mean": np.mean(stereotype_scores),
        "stereotype_std": np.std(stereotype_scores, ddof=1),
        "stereotype_n": len(stereotype_scores),
        "anti_stereotype_mean": np.mean(anti_stereotype_scores),
        "anti_stereotype_std": np.std(anti_stereotype_scores, ddof=1),
        "anti_stereotype_n": len(anti_stereotype_scores),
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
        "is_significant": bool(p_value < alpha)
    }

def process_json_files(input_dir, output_dir):
    """Process each JSON file, detect unique bias types, and perform t-tests for each type."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)

                # Detect unique bias types in the data
                unique_bias_types = set(item.get("bias_type") for item in data if item.get("bias_type"))

                group_results = {}
                for bias_type in unique_bias_types:
                    stats_dict = calculate_group_statistics(data, bias_type)
                    
                    if stats_dict:  # Ensure statistics were successfully calculated
                        result = perform_ttest(
                            stats_dict["stereotype_mean"], stats_dict["anti_stereotype_mean"],
                            stats_dict["stereotype_std"], stats_dict["anti_stereotype_std"],
                            stats_dict["stereotype_n"], stats_dict["anti_stereotype_n"]
                        )
                        group_results[bias_type] = {"statistics": stats_dict, "t_test_results": result}

                with open(output_path, "w") as f:
                    json.dump(group_results, f, indent=4)

                print(f"Processed: {filename} -> {output_path}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = "predictions/stereoset"
output_directory = "output_updated/stereoset/lmb/per_model_scores"
process_json_files(input_directory, output_directory)