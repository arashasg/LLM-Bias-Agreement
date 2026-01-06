# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import numpy as np
# from tqdm import tqdm
# from resources import TOKEN
# import os
# import json
# import dataloader as dataloader # Keeping your import as provided
# import math

# class RedditModel(object):
#     def __init__(self, model_path:str, 
#                  tokenizer_path=None, 
#                  multi_gpu=True, 
#                  input_file="data/sampled_dev.json", 
#                  output_path="./predictions/") -> None:
        
#         self.model_path = model_path
#         self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
#         self.multi_gpu = multi_gpu
        
#         # 1. Setup Output Paths Early
#         model_name = self.model_path.split("/")[-1]
#         self.output_filename = f"{model_name}_perplexity_scores.json"
#         self.output_path = output_path
#         self.full_output_path = os.path.join(self.output_path, self.output_filename)

#         # 2. Initialize Model (Auto-shard across GPUs)
#         self.model = self.initialize_model() 
        
#         self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
#         self.input_file_abs_path = os.path.abspath(input_file)
        
#         self.dataloader = dataloader.LMB(location=input_file)  
#         self.samples = self.dataloader.get_samples()      

#         # 3. Load Checkpoint (Resume if file exists)
#         self.load_checkpoint()

#     def initialize_model(self):
#         """
#         Initialize the model with device_map='auto' to split across GPUs.
#         """
#         print(f"Loading model across {torch.cuda.device_count()} GPUs...")
        
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             token=TOKEN,
#             torch_dtype=torch.bfloat16,
#             # attn_implementation="flash_attention_2", # Comment out if Llama 4 issues arise
#             trust_remote_code=True,
#             cache_dir="/datasets/llms/",
#             # KEY FIX for OOM:
#             device_map="auto" 
#         )
#         return model
    
#     def load_checkpoint(self):
#         """
#         Checks if output file exists. If so, loads previous scores into samples.
#         """
#         if os.path.exists(self.full_output_path):
#             print(f"Found existing output file at {self.full_output_path}. Resuming...")
#             try:
#                 with open(self.full_output_path, "r") as f:
#                     existing_data = json.load(f)
                
#                 # Create a lookup dictionary: Key = sentence
#                 # Using 'sentence' as the unique key based on your data structure
#                 lookup = {item['sentence']: item for item in existing_data}
                
#                 loaded_count = 0
#                 for sample in self.samples:
#                     if sample.sentence in lookup:
#                         data = lookup[sample.sentence]
                        
#                         score = data.get('sentence_perplexity_score')
                        
#                         if score is not None:
#                             sample.sentence_perplexity_score = score
#                             loaded_count += 1
                            
#                 print(f"Successfully loaded {loaded_count} previously calculated samples.")
#             except json.JSONDecodeError:
#                 print("Output file exists but is corrupted. Starting from scratch.")
#         else:
#             print("No existing output file found. Starting fresh.")

#     def perplexity_score(self, sentence):
#         """
#         Finds perplexity score of a sentence.
#         """
#         tokenize_input = self.tokenizer(sentence, return_tensors='pt')
        
#         # Move inputs to the same device as the model's first layer
#         input_ids = tokenize_input['input_ids'].to(self.model.device)
        
#         self.model.eval()

#         with torch.no_grad():
#             # Do NOT move self.model here (breaks sharding)
#             outputs = self.model(input_ids, labels=input_ids)
#             loss = outputs.loss
#             perplexity = math.exp(loss.item())
        
#         del outputs
#         del input_ids
        
#         return perplexity

#     def add_perplexity_scores(self):
#         save_frequency = 50 
        
#         for i, sample in enumerate(tqdm(self.samples, desc="Processing Holistic samples", unit="sample")):
            
#             # Resume Logic: Skip if already calculated
#             if getattr(sample, 'sentence_perplexity_score', None) is not None:
#                 continue

#             # Calculate scores
#             sample.sentence_perplexity_score = self.perplexity_score(sample.sentence)
            
#             # Note: You have replaced_sentence commented out in your original code, 
#             # so I left it out here. If you need it, uncomment below:
#             # sample.replaced_sentence_perplexity_score = self.perplexity_score(sample.replaced_sentence)

#             # Periodic Save
#             if (i + 1) % save_frequency == 0:
#                 self.store_samples_as_json()

#     def store_samples_as_json(self):
#         """
#         Stores the samples using atomic writing to prevent corruption.
#         """
#         output_data = []
#         for sample in self.samples:
#             output_data.append({
#                 "sentence": sample.sentence,
#                 # Holistic specific fields
#                 "axis": getattr(sample, 'axis', None),
#                 "descriptor": getattr(sample, 'descriptor', None),
#                 "gender": getattr(sample, 'gender', None),
                
#                 # Scores
#                 "sentence_perplexity_score": getattr(sample, 'sentence_perplexity_score', None),
#             })

#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)

#         # Atomic Write
#         temp_file = self.full_output_path + ".tmp"
#         with open(temp_file, "w") as f:
#             json.dump(output_data, f, indent=4)
        
#         os.replace(temp_file, self.full_output_path)
    
#     def run(self):
#         self.add_perplexity_scores()
#         self.store_samples_as_json()
#         print(f"Output stored in {self.full_output_path}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader as dataloader
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

        # 2. Initialize Model
        self.model = self.initialize_model() 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        self.input_file_abs_path = os.path.abspath(input_file)
        
        # Note: Your dataloader now handles shuffling, so different jobs likely start at different places
        self.dataloader = dataloader.LMB(location=input_file)  
        self.samples = self.dataloader.get_samples()      

        # 3. Load Checkpoint (Initial load)
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
            trust_remote_code=True,
            cache_dir="/datasets/llms/",
            device_map="auto" 
        )
        return model
    
    def load_checkpoint(self):
        """
        Initial load of the file to populate memory before starting.
        """
        if os.path.exists(self.full_output_path):
            print(f"Found existing output file at {self.full_output_path}. Resuming...")
            try:
                with open(self.full_output_path, "r") as f:
                    existing_data = json.load(f)
                
                # Create a lookup dictionary: Key = sentence
                lookup = {item['sentence']: item.get('sentence_perplexity_score') for item in existing_data}
                
                loaded_count = 0
                for sample in self.samples:
                    if sample.sentence in lookup and lookup[sample.sentence] is not None:
                        sample.sentence_perplexity_score = lookup[sample.sentence]
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
        input_ids = tokenize_input['input_ids'].to(self.model.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        
        del outputs
        del input_ids
        return perplexity

    def add_perplexity_scores(self):
        save_frequency = 50 
        
        for i, sample in enumerate(tqdm(self.samples, desc="Processing Holistic samples", unit="sample")):
            
            # --- CONCURRENCY CHECK ---
            # Even if our memory says None, another job might have finished it since we started.
            # However, checking the file every iteration is too slow. 
            # We rely on 'store_samples_as_json' to sync, or if we want to be safe,
            # we check if we already have it in memory first.
            if getattr(sample, 'sentence_perplexity_score', None) is not None:
                continue

            # Calculate scores
            sample.sentence_perplexity_score = self.perplexity_score(sample.sentence)
            
            # Periodic Save
            if (i + 1) % save_frequency == 0:
                self.store_samples_as_json()

    def store_samples_as_json(self):
        """
        Concurrency-Safe Save:
        1. Reads the file from disk (to get updates from other jobs).
        2. Merges disk data with local memory (local memory takes precedence if we just calculated it).
        3. Updates local memory with disk data (so we don't re-calculate stuff other jobs did).
        4. Writes back to disk.
        """
        
        # 1. READ CURRENT DISK STATE
        disk_scores = {}
        if os.path.exists(self.full_output_path):
            try:
                with open(self.full_output_path, "r") as f:
                    disk_data = json.load(f)
                    # Map sentence -> score
                    for item in disk_data:
                        if item.get('sentence_perplexity_score') is not None:
                            disk_scores[item['sentence']] = item['sentence_perplexity_score']
            except (json.JSONDecodeError, IOError):
                pass # If read fails, we proceed with local data only
        
        output_data = []
        
        # 2. MERGE LOGIC
        for sample in self.samples:
            
            local_score = getattr(sample, 'sentence_perplexity_score', None)
            
            # Case A: We calculated it locally -> Keep local
            final_score = local_score
            
            # Case B: We haven't calculated it, but disk has it -> Use disk & Update local memory
            if final_score is None and sample.sentence in disk_scores:
                final_score = disk_scores[sample.sentence]
                # Sync local memory so we skip this in the main loop later
                sample.sentence_perplexity_score = final_score
            
            # Prepare entry
            entry = {
                "sentence": sample.sentence,
                "axis": getattr(sample, 'axis', None),
                "descriptor": getattr(sample, 'descriptor', None),
                "gender": getattr(sample, 'gender', None),
                "sentence_perplexity_score": final_score
            }
            output_data.append(entry)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # 3. ATOMIC WRITE
        temp_file = self.full_output_path + f".tmp_{os.getpid()}" # Add PID to temp file to avoid temp collisions
        try:
            with open(temp_file, "w") as f:
                json.dump(output_data, f, indent=4)
            os.replace(temp_file, self.full_output_path)
        except OSError as e:
            print(f"Warning: Could not save file due to race condition or lock: {e}")

    def run(self):
        self.add_perplexity_scores()
        self.store_samples_as_json()
        print(f"Output stored in {self.full_output_path}")