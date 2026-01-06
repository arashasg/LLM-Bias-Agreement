import json
import os
from tqdm import tqdm
from honest import honest

def add_mask_token_to_prompts(prompts):
    """
    Appends the [M] token to the end of each prompt.
    """
    return [str(prompt).strip() + " [M]" for prompt in prompts]

def process_file(predictions_file, input_path, output_path, evaluator):
    """
    Reads a prediction file, extracts templates from 'bias_type' and completions 
    from 'generations', and calculates the overall HONEST score.
    """
    # 1. Extract model name from filename
    base_name = predictions_file
    if "_temprature" in base_name:
        model_name = base_name.split("_temprature")[0]
    elif "_perplexity" in base_name:
        model_name = base_name.split("_perplexity")[0]
    else:
        model_name = base_name.replace(".json", "")
        
    output_filename = f"{model_name}.json"
    full_input_path = os.path.join(input_path, predictions_file)
    full_output_path = os.path.join(output_path, output_filename)

    try:
        with open(full_input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        all_templates = []
        all_completions = []
        
        # 2. Iterate through data to collect all samples
        for entry in data:
            # User clarification: 'bias_type' contains the template text
            template_text = entry.get('bias_type', '')
            
            # User clarification: 'generations' contains the list of completion strings
            completions = entry.get('generations', [])

            if template_text and completions:
                all_templates.append(template_text)
                all_completions.append(completions)

        # 3. Calculate Overall Score
        if all_templates:
            # Add [M] token to templates for HONEST
            masked_templates_list = add_mask_token_to_prompts(all_templates)
            
            # Create the dictionary structure HONEST expects: {index: {"template": "..."}}
            masked_templates_dict = {i: {"template": t} for i, t in enumerate(masked_templates_list)}
            
            # Calculate overall score across ALL samples in the file
            # all_completions is a list of lists of strings
            overall_score, _ = evaluator.honest_dataframe(all_completions, masked_templates_dict)
            total_samples = len(all_templates)
        else:
            overall_score = 0.0
            total_samples = 0

        # 4. Save results
        final_output = {
            "model_name": model_name,
            "honest_score": overall_score,
            "number_of_samples": total_samples
        }
        
        with open(full_output_path, 'w', encoding='utf-8') as outfile:
            json.dump(final_output, outfile, indent=4)
            
    except Exception as e:
        print(f"Failed to process {predictions_file}: {e}")

def process_folder(folder_path, output_path):
    """ Processes all JSON files in the folder. """
    if not os.path.exists(folder_path):
        print(f"Error: Input folder '{folder_path}' does not exist.")
        return

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print("No JSON files found in the folder.")
        return

    print("Initializing HONEST evaluator...")
    # Initialize evaluator once
    evaluator = honest.HonestEvaluator(language='en')

    # Iterate over files
    for json_file in tqdm(json_files, desc="Processing files", unit="file"):
        process_file(json_file, folder_path, output_path, evaluator)

if __name__ == "__main__":
    # Configuration
    input_path = "predictions"
    output_path = "honest_result"
    
    os.makedirs(output_path, exist_ok=True)
    
    process_folder(input_path, output_path)