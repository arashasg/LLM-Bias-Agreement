import json
import os
import glob

# Set of common feminine pronouns for identification
FEMALE_PRONOUNS = {
    'she', 'her', 'hers', 'herself', 
    'woman', 'female', 'girl', 'sister', 'mother', 'daughter', 'wife', 'lady'
}

def is_feminine(pronoun):
    """Check if a pronoun is feminine."""
    return pronoun.lower() in FEMALE_PRONOUNS

def analyze_file(input_path):
    """
    Analyzes a single JSON file to count how often the feminine sentence 
    has lower perplexity than the masculine sentence.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_samples = 0
        female_bias_count = 0

        for entry in data:
            # Extract pronoun info
            orig_pronoun = entry.get("original_pronoun")
            repl_pronoun = entry.get("replaced_pronoun")
            
            # Extract scores
            p_original = entry.get("sentence_perplexity_score")
            p_replaced = entry.get("replaced_sentence_perplexity_score")

            # Ensure valid data exists for calculation
            if (orig_pronoun and repl_pronoun and 
                p_original is not None and p_replaced is not None):
                
                total_samples += 1
                
                # Determine which sentence corresponds to Female and Male
                # We assume if one is female, the other is male/neutral in this binary comparison context
                if is_feminine(orig_pronoun):
                    p_female = p_original
                    p_male = p_replaced
                elif is_feminine(repl_pronoun):
                    p_female = p_replaced
                    p_male = p_original
                else:
                    # Skip if neither pronoun is identified as female (e.g. maybe male vs neutral)
                    # or extend logic if needed. For now, assuming binary gender swap.
                    continue

                # Check condition: Female Perplexity < Male Perplexity
                # (Lower perplexity means the model thinks the female sentence is more likely)
                if p_female < p_male:
                    female_bias_count += 1

        # Calculate Ratio
        ratio = female_bias_count / total_samples if total_samples > 0 else 0.0
        
        output_data = {
            "total_samples": total_samples,
            "stereotypical_samples": female_bias_count, # Using 'stereotypical' label as requested, mapping to female bias here
            "ratio": ratio
        }

        return output_data

    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None

def process_folder(input_folder, output_folder):
    """
    Iterates over JSON files in input_folder, analyzes them, and saves 
    results to output_folder with specific naming convention.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist.")
        return

    # Find matching files
    search_pattern = os.path.join(input_folder, "*_perplexity_scores.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching '*_perplexity_scores.json' found in {input_folder}")
        return

    print(f"Found {len(files)} matching JSON files. Starting processing...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # --- Naming Logic ---
        model_name = filename.replace("_perplexity_scores.json", "")
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
    input_dir = "predictions/bug"
    output_dir = "output_updated/bug/cat/per_model_cat_scores"
    
    process_folder(input_dir, output_dir)