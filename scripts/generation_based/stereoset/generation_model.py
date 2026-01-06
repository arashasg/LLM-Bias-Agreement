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
                 input_file="data/sampled_dev.json", 
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
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
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
        clusters = self.dataloader.get_intrasentence_examples()
        if self.results_file_exists(clusters=clusters):
            print("result already exists!")
            return
        self.model.eval()
        
        predictions = self.generate_predictions(clusters)
        # self.store_predictions(predictions)

    
    def generate_intersentence(self)->None:
        clusters = self.dataloader.get_intersentence_examples()
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
                "ID": example.ID,
                "bias_type": example.bias_type,
                "target": example.target,
                "context": example.context,
                "sentences": [
                    {
                        "ID": sentence.ID,
                        "sentence": sentence.sentence,
                        "template_word": sentence.template_word,
                        "gold_label": sentence.gold_label,
                        "completions": getattr(sentence, 'completions', [])  # Use an array for completions
                    }
                    for sentence in example.sentences
                ]
            }
            formatted_predictions.append(formatted_example)
        
        return formatted_predictions

    def store_predictions(self, predictions):
        # Define the output filename based on model parameters
        output_filename = self.get_output_file_name()
        output_file = os.path.join(self.output_path, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save predictions to the JSON file (overwriting each time)
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        print(f"Predictions stored in: {output_file}")
    

    def get_prediction(self, batch, num_generations=25):
        templates = []
        sentence_map = []  # This will hold tuples of (Sentence object, generated completion)

        # Collect all sentences and keep track of which sentence corresponds to which completion
        for sample in batch:
            for sentence in sample.sentences:
                templates.append(self.tokenizer.eos_token + sample.context + " " + sentence.sentence)  # Collect the text for generation
                sentence_map.append(sentence)  # Keep track of the corresponding Sentence object
        
        
        # Tokenize the inputs
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
        # Generate sequences for the entire batch in parallel, with multiple generations per input
        output_ids = self.model.generate(
            **inputs,
            max_length=self.max_len,
            num_return_sequences=num_generations,  # Generate multiple completions per sentence
            do_sample=True,  # Enable sampling to generate diverse completions
            top_p=self.top_p,
            temperature=self.temperature,
            use_cache=False # FIX: Disabled cache to prevent DynamicCache attribute error
        )


        # Adjust output_ids to handle multiple generations per input
        total_sentences = len(sentence_map)
        completions = []
        
        # Decode the generated sequences
        for i in range(total_sentences):
            sentence_completions = [
                self.tokenizer.decode(output_ids[i * num_generations + j], skip_special_tokens=True)
                for j in range(num_generations)
            ]
            completions.append(sentence_completions)
        del output_ids
        # Pair each generated set of completions with its corresponding Sentence object
        for i, sentence_completions in enumerate(completions):
            sentence_map[i].completions = sentence_completions  # Add a 'completions' attribute with multiple completions
        torch.cuda.empty_cache()  
        # At this point, each sentence object in the batch will have a 'completions' attribute with its generated texts.
        return batch  # Return the updated batch with multiple completions attached to sentences

