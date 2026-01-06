import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader_reddit as dataloader
import math
import sys
import gc

class RedditModel(object):
    def __init__(self, model_path:str, 
                 tokenizer_path=None, 
                 multi_gpu=True, 
                 input_file="data/sampled_dev.json", 
                 output_path="./predictions/") -> None:
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.multi_gpu = multi_gpu
        
        # 1. Setup Output Paths
        model_name = self.model_path.split("/")[-1]
        self.output_filename = f"{model_name}_perplexity_scores.json"
        self.output_path = output_path
        self.full_output_path = os.path.join(self.output_path, self.output_filename)

        # 2. Initialize Model
        self.model = self.initialize_model()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.input_file_abs_path = os.path.abspath(input_file)
        
        # 3. Load Data
        self.dataloader = dataloader.LMB(location=input_file)  
        self.samples = self.dataloader.get_samples() 
        
        # 4. Load Checkpoint
        self.load_checkpoint()     

    def initialize_model(self):
        print(f"Loading model across {torch.cuda.device_count()} GPUs...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            token=TOKEN,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir="/datasets/llms/",
            device_map="auto" 
        )
        return model

    def load_checkpoint(self):
        """
        Loads previous scores. Uses INDEX alignment to handle duplicate sentences.
        """
        if os.path.exists(self.full_output_path):
            print(f"Found existing output file at {self.full_output_path}. Resuming...")
            try:
                with open(self.full_output_path, "r") as f:
                    existing_data = json.load(f)
                
                # Truncate if mismatch (e.g. if dataset size changed)
                if len(existing_data) > len(self.samples):
                    existing_data = existing_data[:len(self.samples)]

                loaded_count = 0
                
                # Iterate by INDEX to safely handle duplicate sentences
                for i in range(len(existing_data)):
                    data = existing_data[i]
                    sample = self.samples[i]

                    # Sanity check: Ensure the sentences match
                    if data.get('original_sentence') != sample.original_sentence:
                        print(f"Mismatch at index {i}. Stopping load.")
                        break

                    # Check for VALID scores
                    s_score = data.get('sentence_perplexity_score') or data.get('original_sentence_perplexity_score')
                    
                    if s_score is not None:
                        sample.sentence_perplexity_score = s_score
                        sample.replaced_sentence_perplexity_score = data.get('replaced_sentence_perplexity_score')
                        loaded_count += 1
                            
                print(f"Successfully loaded {loaded_count} valid calculated samples.")
            except json.JSONDecodeError:
                print("Output file exists but is corrupted. Starting from scratch.")
        else:
            print("No existing output file found. Starting fresh.")
    
    def perplexity_score(self, sentence):
        """
        Finds perplexity score. 
        No try/except here so exceptions propagate up (Fail Loudly).
        """
        if not sentence or not isinstance(sentence, str) or len(sentence.strip()) == 0:
            return None

        # Tokenize
        tokenize_input = self.tokenizer(
            sentence, 
            return_tensors='pt', 
            truncation=True, 
            max_length=4096 # Truncate to prevent OOM
        )
        
        input_ids = tokenize_input['input_ids'].to(self.model.device)
        attention_mask = tokenize_input['attention_mask'].to(self.model.device)
        
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if torch.isnan(loss):
                raise ValueError(f"Model returned NaN loss for sentence: {sentence[:50]}...")
            
            perplexity = math.exp(loss.item())
        
        del outputs
        del input_ids
        del attention_mask
        return perplexity
    
    def add_perplexity_scores(self):
        save_frequency = 10 
        
        try:
            for i, sample in enumerate(tqdm(self.samples, desc="Processing samples", unit="sample")):
                
                # Skip if already calculated
                if getattr(sample, 'sentence_perplexity_score', None) is not None:
                    continue

                # Periodic Garbage Collection
                if i % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Calculate scores
                sample.sentence_perplexity_score = self.perplexity_score(sample.original_sentence)
                sample.replaced_sentence_perplexity_score = self.perplexity_score(sample.replaced_sentence)

                # Periodic Save
                if (i + 1) % save_frequency == 0:
                    self.store_samples_as_json()
                    
        except KeyboardInterrupt:
            print("\nRun interrupted by user. Saving progress...")
            self.store_samples_as_json()
            raise
        except Exception as e:
            print(f"\nRun CRASHED at sample {i} with error: {e}")
            print("Saving progress before exiting...")
            self.store_samples_as_json()
            raise e

    def store_samples_as_json(self):
        """
        Stores ONLY the samples that have been processed to avoid 'null' tail.
        """
        output_data = []
        for sample in self.samples:
            
            # STOP saving if we hit a sample that hasn't been processed yet.
            # This prevents filling the file with thousands of 'null' entries at the end.
            if getattr(sample, 'sentence_perplexity_score', None) is None:
                break

            output_data.append({
                "original_sentence": sample.original_sentence,
                "replaced_sentence": sample.replaced_sentence,
                "group": sample.group,
                "original_sentence_perplexity_score": getattr(sample, 'sentence_perplexity_score', None),
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