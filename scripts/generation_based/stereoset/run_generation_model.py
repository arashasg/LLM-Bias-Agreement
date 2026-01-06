import yaml
from generation_model import StereosetModel
import shutil
import os  
import argparse
import torch
# from ollama_generation_model import StereosetModel
# Print the version of PyTorch
print("PyTorch version:", torch.__version__)

# Check if CUDA is available and print the CUDA version
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available.")


def delete_model_folders(path="../.cache/huggingface/hub/"):
    # Check if the provided path exists
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return
    
    # List all directories in the given path
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        # Check if the folder starts with 'models--' and is indeed a directory
        if folder_name.startswith('models--') and os.path.isdir(folder_path):
            print(f"Deleting folder: {folder_name}")
            try:
                shutil.rmtree(folder_path)  # Remove the folder and all its contents
            except Exception as e:
                print(f"Error deleting {folder_name}: {e}")
parser = argparse.ArgumentParser("Arguments required for running the models...")
parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Model name!"
)
args = parser.parse_args()
model_name = args.model

# Load the YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Temperature and top_p (nucleus sampling) combinations
temperature_values = [1.0, 0.7, 1.2, 1.5]  # Includes the default (1.0)
top_p_values = [1.0, 0.8, 0.9, 0.7]  # Includes the default (1.0)

# temperature_values = [0.7, 0.9, 1.0, 1.2, 1.5]  # Includes the default (1.0)
# top_p_values = [0.85, 0.9, 0.95, 1.0, 0.7]  # Includes the default (1.0)

# Max token length options
max_lengths = [300, 450, 600]


# Loop over the different combinations of temperature, top_p, and max_len
for temperature in temperature_values:
    for top_p in top_p_values:
        for max_len in max_lengths:
            
            # Initialize the StereosetModel with the current combination
            stereoset_model = StereosetModel(
                model_path=model_name,
                top_p=top_p,
                temperature=temperature,
                multi_gpu=True,
                max_len=max_len
            )
            
            # Output the current combination being tested
            print(f"Running for model: {model_name}, "
                    f"Temperature: {temperature}, Top_p: {top_p}, Max Length: {max_len}")
            
            # Run the intrasentence generation for the current configuration
            stereoset_model.generate_intersentence()
delete_model_folders()

