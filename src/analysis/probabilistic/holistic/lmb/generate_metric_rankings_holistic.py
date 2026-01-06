import os
import json

def generate_rankings():
    # Hardcoded input and output paths
    input_dir = "output_updated/holistic/student_ttest_results"
    output_file = "output_updated/holistic/lmb/holistic_metric_rankings.json"
    
    # Create output directory if it doesn't exist
    output_dir_path = os.path.dirname(output_file)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        
    all_rankings = {}
    
    # Mapping from dataset names to image names
    axis_mapping = {
        "sexual_orientation": "sexual-orientation",
        "age": "age",
        "religion": "religion",
        "nationality": "nationality",
        "race_ethnicity": "race-color",
        "socioeconomic_class": "socioeconomic",
        "gender_and_sex": "gender",
        "body_type": "physical-appearance",
        "ability": "disability",            # New mapping
        "political_ideologies": "profession" # New mapping
    }

    # Axes to include (based on the image + mapped names)
    included_axes = [
        "sexual-orientation",
        "age",
        "religion",
        "nationality",
        "race-color",
        "socioeconomic",
        "gender",
        "physical-appearance",
        "disability",
        "profession"
    ]
    
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                # Extract model name: everything before "_info"
                # Example: "Meta-Llama-3-8B-Instruct_info.json" -> "Meta-Llama-3-8B-Instruct"
                model_name = filename.split('_info')[0]
                
                filepath = os.path.join(input_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    temp_scores = []
                    for axis, details in data.items():
                        # Check if axis is in our mapping
                        mapped_axis = axis_mapping.get(axis)
                        
                        if mapped_axis and mapped_axis in included_axes:
                            stats = details.get('statistics')
                            if stats:
                                top_mean = stats.get('top_5_mean')
                                last_mean = stats.get('last_5_mean')
                                
                                # Ensure both values exist before calculation
                                if top_mean is not None and last_mean is not None:
                                    # Calculate the difference
                                    diff = last_mean - top_mean
                                    temp_scores.append({
                                        "axis": mapped_axis,
                                        "diff": diff
                                    })
                    
                    # Sort by difference in descending order (highest difference first)
                    temp_scores.sort(key=lambda x: x["diff"], reverse=True)
                    
                    # Add ranking to the data
                    ranked_model_data = []
                    for rank, item in enumerate(temp_scores, 1):
                        ranked_model_data.append({
                            "rank": rank,
                            "axis": item["axis"],
                            "diff": item["diff"]
                        })
                    
                    all_rankings[model_name] = ranked_model_data
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    else:
        print(f"Input directory {input_dir} not found.")
    
    # Write to single output JSON file
    with open(output_file, 'w') as f:
        json.dump(all_rankings, f, indent=4)
        
    print(f"Rankings saved to {output_file}")

if __name__ == "__main__":
    generate_rankings()