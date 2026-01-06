# import torch.version
import yaml
from generation_model_bug import RedditModel
import shutil
import os  
import torch
import sys
from resources import data_paths


# Print the version of PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"Cuda version: {torch.version.cuda}")
current_dataset = "bug"
# current_dataset = "holistic"

def create_directory(path):
    try:
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)
        print(f"Directory created at: {path}")
    except Exception as e:
        print(f"Error creating directory at {path}: {e}")

def delete_model_folders(path="../../.cache/huggingface/hub/"):
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    input_model_name = sys.argv[1]

    # Load the YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(input_model_name)
    # Check if the specified model exists in the config
    model_config = next((model for model in config['models'] if model['name'] == input_model_name), None)
    print(model_config)
    if not model_config:
        print(f"Model '{input_model_name}' not found in the configuration.")
        sys.exit(1)

    # Run the model if found
    use_multi_gpu = model_config['use_multi_gpu']
    output_path = os.path.join("./predictions/",current_dataset)
    create_directory(output_path)
    model = RedditModel(input_model_name, multi_gpu=use_multi_gpu, input_file=data_paths[current_dataset], output_path=output_path)
    model.run()
    # delete_model_folders()

if __name__ == "__main__":

    main()
