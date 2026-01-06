import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader_crows as dataloader
import math

class RedditModel(object):
    def __init__(self, model_path:str, 
                 tokenizer_path=None, 
                 multi_gpu=True, 
                 input_file="data/sampled_dev.json", 
                 output_path="./predictions/") -> None:
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.multi_gpu = multi_gpu
        
        # 1. Setup Output Paths Early
        model_name = self.model_path.split("/")[-1]
        self.output_filename = f"{model_name}_perplexity_scores.json"
        self.output_path = output_path
        self.full_output_path = os.path.join(self.output_path, self.output_filename)

        # 2. Initialize Model (Auto-shard across GPUs)
        self.model = self.initialize_model() 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        self.input_file_abs_path = os.path.abspath(input_file)
        
        self.dataloader = dataloader.LMB(location=input_file)  
        self.samples = self.dataloader.get_samples()      

        # 3. Load Checkpoint (Resume if file exists)
        self.load_checkpoint()

    def initialize_model(self):
        """
        Initialize the model with device_map='auto' to split across GPUs.
        """
        print(f"Loading model across {torch.cuda.device_count()} GPUs...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            token=TOKEN,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
            trust_remote_code=True,
            # cache_dir="/datasets/llms/",
            # KEY FIX for OOM:
            device_map="auto" 
        )
        return model
    
    def load_checkpoint(self):
        """
        Checks if output file exists. If so, loads previous scores into samples.
        """
        if os.path.exists(self.full_output_path):
            print(f"Found existing output file at {self.full_output_path}. Resuming...")
            try:
                with open(self.full_output_path, "r") as f:
                    existing_data = json.load(f)
                
                # Create a lookup dictionary: Key = original_sentence
                lookup = {item['original_sentence']: item for item in existing_data}
                
                loaded_count = 0
                for sample in self.samples:
                    if sample.original_sentence in lookup:
                        data = lookup[sample.original_sentence]
                        
                        # Support both naming conventions just in case
                        orig_score = data.get('sentence_perplexity_score') or data.get('original_sentence_perplexity_score')
                        rep_score = data.get('replaced_sentence_perplexity_score')
                        
                        if orig_score is not None:
                            sample.sentence_perplexity_score = orig_score
                        if rep_score is not None:
                            sample.replaced_sentence_perplexity_score = rep_score
                            
                        if orig_score is not None:
                            loaded_count += 1
                            
                print(f"Successfully loaded {loaded_count} previously calculated samples.")
            except json.JSONDecodeError:
                print("Output file exists but is corrupted. Starting from scratch.")
        else:
            print("No existing output file found. Starting fresh.")

    def perplexity_score(self, sentence):
        """
        Finds perplexity score of a sentence.
        """
        tokenize_input = self.tokenizer(sentence, return_tensors='pt')
        
        # Move inputs to the same device as the model's first layer
        input_ids = tokenize_input['input_ids'].to(self.model.device)
        
        self.model.eval()

        with torch.no_grad():
            # Do NOT move self.model here (breaks sharding)
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        
        del outputs
        del input_ids
        
        return perplexity

    def add_perplexity_scores(self):
        save_frequency = 50 
        
        for i, sample in enumerate(tqdm(self.samples, desc="Processing Crows samples", unit="sample")):
            
            # Resume Logic: Skip if already calculated
            if getattr(sample, 'sentence_perplexity_score', None) is not None:
                continue

            # Calculate scores
            sample.sentence_perplexity_score = self.perplexity_score(sample.original_sentence)
            sample.replaced_sentence_perplexity_score = self.perplexity_score(sample.replaced_sentence)
            
            # Periodic Save
            if (i + 1) % save_frequency == 0:
                self.store_samples_as_json()

    def store_samples_as_json(self):
        """
        Stores the samples using atomic writing to prevent corruption.
        """
        output_data = []
        for sample in self.samples:
            output_data.append({
                "original_sentence": sample.original_sentence,
                "replaced_sentence": sample.replaced_sentence,
                
                # Optional fields (using getattr to be safe if dataloader changes)
                "bias_type": getattr(sample, 'bias_type', None), 
                "group": getattr(sample, 'group', None),
                
                # Scores
                "sentence_perplexity_score": getattr(sample, 'sentence_perplexity_score', None),
                "replaced_sentence_perplexity_score": getattr(sample, 'replaced_sentence_perplexity_score', None)
            })

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Atomic Write
        temp_file = self.full_output_path + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(output_data, f, indent=4)
        
        os.replace(temp_file, self.full_output_path)
    
    def run(self):
        self.add_perplexity_scores()
        self.store_samples_as_json()
        print(f"Output stored in {self.full_output_path}")