import os
import json
import numpy as np
from scipy import stats

def calculate_statistics_for_subset(data, filter_key, filter_value):
    """
    Calculate mean and standard deviation for original and replaced 
    perplexity scores for a subset of data defined by a key-value pair.
    """
    
    # Filter using list comprehension with safe .get()
    subset = [item for item in data if item.get(filter_key) == filter_value]
    
    # Robustly get scores, handling potential missing keys
    original_scores = []
    replaced_scores = []
    
    for item in subset:
        # Check for standard key names
        orig = item.get("sentence_perplexity_score")
        rep = item.get("replaced_sentence_perplexity_score")
        
        # If standard keys missing, try alternative keys
        if orig is None:
            orig = item.get("original_sentence_perplexity_score")
        
        if orig is not None and rep is not None:
            original_scores.append(orig)
            replaced_scores.append(rep)

    if len(original_scores) < 2 or len(replaced_scores) < 2:
        return None  # Skip subsets with insufficient data for t-test

    stats_dict = {
        "original_mean": np.mean(original_scores),
        "original_std": np.std(original_scores, ddof=1),
        "original_n": len(original_scores),
        "replaced_mean": np.mean(replaced_scores),
        "replaced_std": np.std(replaced_scores, ddof=1),
        "replaced_n": len(replaced_scores),
        # Calculate absolute difference for ranking purposes
        "mean_diff": abs(np.mean(original_scores) - np.mean(replaced_scores))
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
    """Process each JSON file, calculate stats for groups and occupations, add rankings, and perform t-tests."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)

                results = {
                    "groups": {},
                    "occupations": {}
                }

                # --- Process Groups ---
                unique_groups = set(item.get("group") for item in data if item.get("group"))
                for group in unique_groups:
                    stats_dict = calculate_statistics_for_subset(data, "group", group)
                    if stats_dict:
                        t_test = perform_ttest(
                            stats_dict["original_mean"], stats_dict["replaced_mean"],
                            stats_dict["original_std"], stats_dict["replaced_std"],
                            stats_dict["original_n"], stats_dict["replaced_n"]
                        )
                        # Remove temporary helper key before saving if desired, or keep it
                        # del stats_dict["mean_diff"] 
                        results["groups"][group] = {"statistics": stats_dict, "t_test_results": t_test}

                # --- Process Occupations & Calculate Rankings ---
                unique_occupations = set(item.get("occupation") for item in data if item.get("occupation"))
                temp_occupation_data = []

                for occupation in unique_occupations:
                    stats_dict = calculate_statistics_for_subset(data, "occupation", occupation)
                    if stats_dict:
                        t_test = perform_ttest(
                            stats_dict["original_mean"], stats_dict["replaced_mean"],
                            stats_dict["original_std"], stats_dict["replaced_std"],
                            stats_dict["original_n"], stats_dict["replaced_n"]
                        )
                        
                        # Store in temp list for sorting
                        temp_occupation_data.append({
                            "name": occupation,
                            "statistics": stats_dict,
                            "t_test_results": t_test,
                            "diff": stats_dict["mean_diff"]
                        })

                # Sort occupations by mean_diff ascending (smaller diff is better/lower rank)
                temp_occupation_data.sort(key=lambda x: x["diff"])

                # Assign ranks and populate final results dictionary
                for rank, item in enumerate(temp_occupation_data, 1):
                    # Add rank to the object
                    item["rank"] = rank
                    # Remove helper key 'diff' to keep output clean (optional)
                    del item["diff"]
                    # Add to results
                    results["occupations"][item["name"]] = {
                        "statistics": item["statistics"],
                        "t_test_results": item["t_test_results"],
                        "rank": item["rank"]
                    }

                # Save results
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"Processed: {filename} -> {output_path}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
if __name__ == "__main__":
    input_directory = "predictions/winobias"
    output_directory = "output_updated/wino/lmb/per_model_scores"
    process_json_files(input_directory, output_directory)