import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader
from tqdm import tqdm
from resources import TOKEN
# import deepspeed
# import torch.distributed as dist

class StereosetModel(object):
    def __init__(self, model_path:str, 
                 tokenizer_path=None, 
                 top_p=1.0, 
                 temperature=1.0, 
                 top_k=50, 
                 multi_gpu=True, 
                 max_len=250,
                 batch_size=4,
                 input_file="data/bold/prompts", 
                 output_path="./predictions/") -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.multi_gpu = multi_gpu
        self.model = self.initialize_model()  # Initialize the model during the class construction
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = max_len
        self.UNCONDITIONAL_START_TOKEN = self.get_unconditional_start_token()
        # self.EOS_TOKEN_ID = self.tokenizer.eos_token_id
        # self.PAD_TOKEN_ID = self.tokenizer.eos_token_id 
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.input_file_abs_path = os.path.abspath(input_file)
        self.output_path = output_path
        self.dataloader = dataloader.StereoSet(location=input_file)
        self.batch_size = batch_size


        


    def initialize_model(self):
        """Function to initialize the model based on multi-GPU or single-GPU configuration."""
    
        # Define the default and fallback paths for the model
        model_path = os.path.join("../../../model-weights/", self.model_path.split("/")[-1])
        
        # Check if the model exists at model_path, otherwise fall back to self.model_path
        if os.path.exists(model_path):
            path_to_load = model_path
            cache_dir = None  # No need for cache if model exists locally
        else:
            path_to_load = self.model_path
            job_id = os.getenv('JOB_ID', 'default_job_id')
            cache_dir = f"/datasets/llms/"  # Set cache path if model is to be downloaded

        print(f"Initializing model across {torch.cuda.device_count()} GPUs...")
        # Distribute the model across all available GPUs using device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            path_to_load,
            token=TOKEN,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # use_flash_attention_2=True,
            attn_implementation="flash_attention_2",
            # attn_implementation="eager",
            cache_dir=cache_dir
        )

        return model


    
    def get_unconditional_start_token(self):
        if self.tokenizer.cls_token is not None:
            return self.tokenizer.cls_token
        elif self.tokenizer.bos_token is not None:
            return self.tokenizer.bos_token
        else:
            return ""

    
    def generate_intrasentence(self)->None:
        clusters = self.dataloader.get_bold_examples()
        if self.results_file_exists(clusters=clusters):
            print("result already exists!")
            return
        self.model.eval()
        
        predictions = self.generate_predictions(clusters)
        # self.store_predictions(predictions)

    
    def generate_intersentence(self)->None:
        clusters = self.dataloader.get_bold_examples()
        if self.results_file_exists(clusters=clusters):
            print("result already exists!")
            return
        self.model.eval()
        
        predictions = self.generate_predictions(clusters)
        # self.store_predictions(predictions)


    def calculate_sentence_score(self, joint_sentence_probability):
        score = np.sum([np.log2(i) for i in joint_sentence_probability]) / len(joint_sentence_probability)
        return np.power(2, score)


    def results_file_exists(self, clusters):
        """Check if the results file exists and contains the correct number of predictions."""
        
        # Define the output filename based on model parameters
        output_filename = self.get_output_file_name()
        output_file = os.path.join(self.output_path, output_filename)
        
        # Check if the output file exists
        if not os.path.exists(output_file):
            return False

        # Load predictions from the file
        with open(output_file, "r") as f:
            predictions = json.load(f)
        
        # Check if the number of predictions matches the number of clusters
        return len(predictions) == len(clusters)
    
    def get_output_file_name(self):
        model_name = os.path.basename(self.model_path)
        return (
            f"{model_name}_temprature_{self.temperature}_nucleus_{self.top_p}_"
            f"maxlen_{self.max_len}_intrasentence_predictions.json"
        )


    

    def generate_predictions(self, clusters):
        predictions = []
        
        # Define the output filename based on model parameters
        output_filename = self.get_output_file_name()
        output_file = os.path.join(self.output_path, output_filename)
        
        # Load existing predictions if the file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                predictions = json.load(f)
            print(f"Loaded existing predictions from {output_file}")
        
        # Determine the starting batch index
        completed_batches = len(predictions) // self.batch_size
        start_index = completed_batches * self.batch_size
        
        # Iterate through remaining batches, starting from the last completed batch
        for i in tqdm(range(start_index, len(clusters), self.batch_size), desc="Generating predictions", unit="batch"):
            batch = clusters[i:i + self.batch_size]
            
            # Generate predictions for the current batch
            batch_predictions = self.get_prediction(batch)
            
            # Format predictions for storage
            formatted_batch_predictions = self.format_predictions(batch_predictions)
            
            # Add formatted predictions to the main list
            predictions.extend(formatted_batch_predictions)
            
            # Save intermediate predictions every 5 batches
            if ((i - start_index) // self.batch_size + 1) % 5 == 0:
                self.store_predictions(predictions=predictions)

        # Save final predictions after completing all batches
        self.store_predictions(predictions=predictions)
        
        return predictions

    def format_predictions(self, batch_predictions):
        formatted_predictions = []
        for example in batch_predictions:
            formatted_example = {
                "ID": example['id'],
                "bias_type": example['type'],
                "sub_type": example["subtype"],
                "focus": example['focus'],
                "text": example['text'],
                "generations": example["generations"]
            }
            formatted_predictions.append(formatted_example)
        
        return formatted_predictions

    def store_predictions(self, predictions):
        # Define the output filename
        output_filename = self.get_output_file_name()
        output_file = os.path.join(self.output_path, output_filename)
        
        # Define a temporary file path
        temp_file = output_file + ".tmp"
        
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        try:
            # 1. Write to a temporary file first (Safe!)
            with open(temp_file, "w") as f:
                json.dump(predictions, f, indent=2)
                f.flush() # Force write to disk
                os.fsync(f.fileno()) # Ensure it's physically written

            # 2. Rename temp file to actual file (Atomic operation)
            # This replaces the old file instantly and is much safer on network drives
            os.replace(temp_file, output_file)
            
            print(f"Predictions stored in: {output_file}")
            
        except OSError as e:
            print(f"Warning: Could not save predictions due to I/O error: {e}")
            # Optional: Don't crash, just continue to the next batch
    

    def get_prediction(self, batch, num_generations=25):
        """
        Generates a specified number of completions for each sample in a batch.

        This function tokenizes the text from each sample, generates completions using
        the model, decodes the results while removing the original prompt text, and
        stores the list of completions in a `generations` attribute on each sample.

        Args:
            batch (list): A list of sample objects, each with a .text attribute.
            num_generations (int): The number of completions to generate for each sample.

        Returns:
            list: The original batch of sample objects, now updated with a 
                `sample.generations` attribute containing the list of completions.
        """
        
        # 1. Prepare prompts from the input batch
        # We create a simple list of strings to feed to the tokenizer.
        prompts = [sample['text'] for sample in batch]
        all_completions = []
        # 2. Tokenize the inputs
        # Ensure the tokenizer has a padding token defined.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        # Store the length of the prompt tokens to remove it from the output later.
        # This is the crucial step to prevent "idle generations".
        input_length = inputs.input_ids.shape[1]

        # 3. Generate sequences for the entire batch
        # The output will be a tensor of shape (batch_size * num_generations, sequence_length)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_len, # Use max_new_tokens to control generation length
            num_return_sequences=num_generations,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
        )

        # 4. Decode the generated sequences, stripping the prompt
        # We decode all generated sequences at once for efficiency.
        # The slice `output_id[input_length:]` removes the prompt tokens.
        decoded_completions = [
            self.tokenizer.decode(output_id[input_length:], skip_special_tokens=True)
            for output_id in output_ids
        ]
        
        # The `decoded_completions` is currently a flat list of size (batch_size * 25).
        # We need to group them into lists of 25.
        
        # 5. Group completions and assign them back to the original samples
        # We iterate through the original batch and the grouped completions simultaneously.
        # The zip function makes this pairing clean and understandable.
        
        # This list comprehension groups the flat list into a list of lists.
        grouped_generations = [
            decoded_completions[i : i + num_generations]
            for i in range(0, len(decoded_completions), num_generations)
        ]

        for sample, generations in zip(batch, grouped_generations):
            # Store the list of 25 completions in the 'generations' attribute.
            sample["generations"] = generations

        # 6. Clean up memory and return the updated batch
        del output_ids
        del decoded_completions
        torch.cuda.empty_cache()
        
        return batch



