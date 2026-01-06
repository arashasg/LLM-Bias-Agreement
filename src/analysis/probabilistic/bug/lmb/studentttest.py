import os
import json
import numpy as np
from scipy import stats

# Set of common feminine pronouns for identification
FEMALE_PRONOUNS = {
    'she', 'her', 'hers', 'herself', 
    'woman', 'female', 'girl', 'sister', 'mother', 'daughter', 'wife', 'lady'
}

def is_feminine(pronoun):
    """Check if a pronoun is feminine."""
    return str(pronoun).lower() in FEMALE_PRONOUNS

def calculate_gender_statistics(data):
    """
    Identifies female vs male sentences based on pronouns and calculates statistics.
    Returns a dictionary with stats for both groups.
    Skips entries where perplexity scores are None or NaN.
    """
    female_scores = []
    male_scores = []

    for item in data:
        # Extract necessary fields
        orig_pronoun = item.get("original_pronoun")
        repl_pronoun = item.get("replaced_pronoun")
        p_original = item.get("sentence_perplexity_score")
        p_replaced = item.get("replaced_sentence_perplexity_score")

        # Validation: Check existence, not None, and not NaN
        # We must check 'is not None' before 'np.isnan' to avoid type errors
        if (orig_pronoun and repl_pronoun and 
            p_original is not None and not np.isnan(p_original) and
            p_replaced is not None and not np.isnan(p_replaced)):
            
            # Determine gender association
            if is_feminine(orig_pronoun):
                female_scores.append(p_original)
                male_scores.append(p_replaced)
            elif is_feminine(repl_pronoun):
                female_scores.append(p_replaced)
                male_scores.append(p_original)
            # If neither or both are female (unlikely in binary setup), we skip

    # Ensure sufficient data for statistics
    if len(female_scores) < 2 or len(male_scores) < 2:
        return None

    stats_dict = {
        "female_mean": np.mean(female_scores),
        "female_std": np.std(female_scores, ddof=1),
        "female_n": len(female_scores),
        "male_mean": np.mean(male_scores),
        "male_std": np.std(male_scores, ddof=1),
        "male_n": len(male_scores),
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
    """Process each JSON file, calculate gender-based perplexity stats, and perform t-test."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(input_path, "r", encoding='utf-8') as f:
                    data = json.load(f)

                # Calculate stats based on gender pronouns
                stats_dict = calculate_gender_statistics(data)
                
                output_results = {}
                
                if stats_dict:
                    # Perform T-test between Female and Male groups
                    result = perform_ttest(
                        stats_dict["female_mean"], stats_dict["male_mean"],
                        stats_dict["female_std"], stats_dict["male_std"],
                        stats_dict["female_n"], stats_dict["male_n"]
                    )
                    
                    output_results = {
                        "statistics": stats_dict,
                        "t_test_results": result
                    }
                else:
                    print(f"Skipping {filename}: Insufficient valid gender data (possibly due to NaNs).")

                # Write results to output file
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(output_results, f, indent=4)

                print(f"Processed: {filename} -> {output_path}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
if __name__ == "__main__":
    # Update paths as needed
    input_directory = "predictions/bug"
    output_directory = "output_updated/bug/lmb/per_model_scores"
    process_json_files(input_directory, output_directory)