import os
import json
import numpy as np
from scipy import stats

def calculate_group_statistics(top_5_scores, last_5_scores):
    """
    Calculate the mean, standard deviation, and number of samples for the top 5 and last 5 groups.

    Parameters:
        top_5_scores (list): List of mean perplexity scores for the top 5 descriptors.
        last_5_scores (list): List of mean perplexity scores for the last 5 descriptors.

    Returns:
        stats_dict (dict): A dictionary containing the mean, standard deviation, and number of samples
                           for both top 5 and last 5 groups. Returns None if data is insufficient.
    """
    # Check if there is enough data to compute statistics
    if not top_5_scores or not last_5_scores:
        return None

    # Calculate mean, standard deviation, and number of samples
    stats_dict = {
        "top_5_mean": np.mean(top_5_scores),
        "top_5_std": np.std(top_5_scores, ddof=1) if len(top_5_scores) > 1 else 0,
        "top_5_n": len(top_5_scores),
        "last_5_mean": np.mean(last_5_scores),
        "last_5_std": np.std(last_5_scores, ddof=1) if len(last_5_scores) > 1 else 0,
        "last_5_n": len(last_5_scores),
    }

    return stats_dict

def perform_ttest(mean1, mean2, std1, std2, n1, n2, alpha=0.05):
    """
    Perform an independent two-sample t-test to determine if the difference between two means is statistically significant.
    """
    # Calculate the t-statistic and p-value using scipy's ttest_ind_from_stats function
    t_stat, p_value = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=n1,
                                                 mean2=mean2, std2=std2, nobs2=n2,
                                                 equal_var=False)  # Assume unequal variances (Welch's t-test)

    # Determine if the result is statistically significant
    is_significant = bool(p_value < alpha)

    # Return results as a dictionary
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant": is_significant
    }

def process_json_files(input_dir, output_dir):
    """
    Process each JSON file, identify top/last 5 descriptors based on perplexity, 
    calculate statistics, and perform t-tests.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r") as f:
                data = json.load(f)

            results = {}
            for category, category_data in data.items():
                # Check if descriptors statistics exist
                if "descriptors_statistics" in category_data:
                    descriptors = category_data["descriptors_statistics"]
                    
                    # Extract all mean_perplexity values
                    # The logic here assumes we want to compare the lowest perplexity (Top) 
                    # vs highest perplexity (Bottom/Last) descriptors.
                    all_scores = []
                    for desc_name, desc_stats in descriptors.items():
                        if "mean_perplexity" in desc_stats:
                            all_scores.append(desc_stats["mean_perplexity"])
                    
                    # Sort scores to identify top 5 and last 5
                    # Sorted ascending: first 5 are lowest (best?), last 5 are highest (worst?)
                    all_scores.sort()
                    
                    if len(all_scores) >= 10:
                        top_5_scores = all_scores[:5]
                        last_5_scores = all_scores[-5:]
                    elif len(all_scores) >= 2:
                        # Fallback for small datasets: split available data
                        mid = len(all_scores) // 2
                        top_5_scores = all_scores[:mid]
                        last_5_scores = all_scores[-mid:]
                    else:
                        top_5_scores = []
                        last_5_scores = []

                    # Calculate statistics
                    stats_dict = calculate_group_statistics(top_5_scores, last_5_scores)

                    if stats_dict:  # Proceed only if there is enough data
                        # Perform t-test
                        result = perform_ttest(
                            mean1=stats_dict["top_5_mean"],
                            mean2=stats_dict["last_5_mean"],
                            std1=stats_dict["top_5_std"],
                            std2=stats_dict["last_5_std"],
                            n1=stats_dict["top_5_n"],
                            n2=stats_dict["last_5_n"]
                        )

                        # Store results
                        results[category] = {
                            "statistics": stats_dict,
                            "t_test_results": result
                        }

            # Save results to output file
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Processed: {filename} -> {output_path}")

# Example usage
input_directory = "holistic_output_info"
output_directory = "output_updated/holistic/student_ttest_results"
process_json_files(input_directory, output_directory)