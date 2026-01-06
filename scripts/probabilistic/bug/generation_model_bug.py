# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import numpy as np
# from tqdm import tqdm
# from resources import TOKEN
# import os
# import json
# import dataloader_bug as dataloader
# import math
# import sys
# import gc

# class RedditModel(object):
#     def __init__(self, model_path:str, 
#                  tokenizer_path=None, 
#                  multi_gpu=True, 
#                  input_file="data/sampled_dev.json", 
#                  output_path="./predictions/") -> None:
        
#         self.model_path = model_path
#         self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
#         self.multi_gpu = multi_gpu
        
#         # 1. Setup Output Paths
#         model_name = self.model_path.split("/")[-1]
#         self.output_filename = f"{model_name}_perplexity_scores.json"
#         self.output_path = output_path
#         self.full_output_path = os.path.join(self.output_path, self.output_filename)

#         # 2. Initialize Model
#         self.model = self.initialize_model()
        
#         self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
#         # Set pad token if missing (common in Llama models)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#         self.input_file_abs_path = os.path.abspath(input_file)
        
#         # 3. Load Data
#         self.dataloader = dataloader.LMB(location=input_file)  
#         self.samples = self.dataloader.get_samples() 
        
#         # 4. Load Checkpoint
#         self.load_checkpoint()     

#     def initialize_model(self):
#         print(f"Loading model across {torch.cuda.device_count()} GPUs...")
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             token=TOKEN,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             cache_dir="/datasets/llms/",
#             device_map="auto" 
#         )
#         return model

#     def load_checkpoint(self):
#         if os.path.exists(self.full_output_path):
#             print(f"Found existing output file at {self.full_output_path}. Resuming...")
#             try:
#                 with open(self.full_output_path, "r") as f:
#                     existing_data = json.load(f)
                
#                 # Check alignment
#                 if len(existing_data) > len(self.samples):
#                     print("Warning: Output file is larger than input dataset. Truncating extra data.")
#                     existing_data = existing_data[:len(self.samples)]

#                 loaded_count = 0
                
#                 # Iterate by INDEX to safely handle duplicate sentences
#                 for i in range(len(existing_data)):
#                     data = existing_data[i]
#                     sample = self.samples[i]

#                     # Sanity check: Ensure the sentences match
#                     if data.get('original_sentence') != sample.original_sentence:
#                         # If mismatch, we stop loading to prevent data corruption
#                         print(f"Mismatch at index {i}. Stopping load.")
#                         break

#                     # Check for VALID scores (not None)
#                     s_score = data.get('sentence_perplexity_score')
                    
#                     if s_score is not None:
#                         sample.sentence_perplexity_score = s_score
#                         sample.replaced_sentence_perplexity_score = data.get('replaced_sentence_perplexity_score')
#                         loaded_count += 1
                            
#                 print(f"Successfully loaded {loaded_count} valid calculated samples.")
#             except json.JSONDecodeError:
#                 print("Output file exists but is corrupted. Starting from scratch.")
#         else:
#             print("No existing output file found. Starting fresh.")
    
#     def perplexity_score(self, sentence):
#         if not sentence or not isinstance(sentence, str) or len(sentence.strip()) == 0:
#             return None

#         try:
#             # FIX: Added truncation and max_length to prevent OOM on long sentences
#             # FIX: Added return_attention_mask=True explicitly
#             tokenize_input = self.tokenizer(
#                 sentence, 
#                 return_tensors='pt', 
#                 truncation=True, 
#                 max_length=2048, # Limit context to prevent OOM
#                 return_attention_mask=True
#             )
            
#             input_ids = tokenize_input['input_ids'].to(self.model.device)
#             attention_mask = tokenize_input['attention_mask'].to(self.model.device)
            
#             self.model.eval()

#             with torch.no_grad():
#                 outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
#                 loss = outputs.loss
                
#                 if torch.isnan(loss):
#                     print(f"NaN loss detected for: {sentence[:30]}...")
#                     return None
                    
#                 perplexity = math.exp(loss.item())
            
#             del outputs
#             del input_ids
#             del attention_mask
#             return perplexity
            
#         except Exception as e:
#             # If OOM happens, we catch it, clear cache, and return None
#             if "out of memory" in str(e).lower():
#                 print(f"OOM Error on sample. Clearing cache.")
#                 torch.cuda.empty_cache()
#             else:
#                 print(f"Error calculating perplexity: {e}")
#             return None
    
#     def add_perplexity_scores(self):
#         save_frequency = 10 
        
#         try:
#             for i, sample in enumerate(tqdm(self.samples, desc="Processing samples", unit="sample")):
                
#                 # Skip if already calculated
#                 if getattr(sample, 'sentence_perplexity_score', None) is not None:
#                     continue

#                 # Periodic Garbage Collection (Crucial for preventing slowdowns)
#                 if i % 50 == 0:
#                     gc.collect()
#                     torch.cuda.empty_cache()

#                 # Calculate scores
#                 sample.sentence_perplexity_score = self.perplexity_score(sample.original_sentence)
#                 sample.replaced_sentence_perplexity_score = self.perplexity_score(sample.replaced_sentence)

#                 # Periodic Save
#                 if (i + 1) % save_frequency == 0:
#                     self.store_samples_as_json()
                    
#         except KeyboardInterrupt:
#             print("\nRun interrupted by user. Saving progress...")
#             raise
#         except Exception as e:
#             print(f"\nRun crashed with error: {e}")
#             raise
#         finally:
#             self.store_samples_as_json()

#     def store_samples_as_json(self):
#         output_data = []
#         for sample in self.samples:
#             output_data.append({
#                 "original_sentence": sample.original_sentence,
#                 "replaced_sentence": sample.replaced_sentence,
#                 "original_pronoun": sample.original_pronoun,
#                 "replaced_pronoun": sample.replaced_pronoun,
#                 "gender": getattr(sample, 'gender', None),
#                 "sentence_perplexity_score": getattr(sample, 'sentence_perplexity_score', None),
#                 "replaced_sentence_perplexity_score": getattr(sample, 'replaced_sentence_perplexity_score', None)
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
#         print(f"Output stored in {self.full_output_path}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader_bug as dataloader
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
        
        model_name = self.model_path.split("/")[-1]
        self.output_filename = f"{model_name}_perplexity_scores.json"
        self.output_path = output_path
        self.full_output_path = os.path.join(self.output_path, self.output_filename)

        # Initialize Model (FP16 as requested)
        self.model = self.initialize_model()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token=TOKEN, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.input_file_abs_path = os.path.abspath(input_file)
        self.dataloader = dataloader.LMB(location=input_file)  
        self.samples = self.dataloader.get_samples() 
        self.load_checkpoint()     

    def initialize_model(self):
        print(f"Loading model across {torch.cuda.device_count()} GPUs...", flush=True)
        local_model_path = os.path.join("../../../model-weights/", self.model_path.split("/")[-1])
        path_to_use = local_model_path if os.path.exists(local_model_path) else self.model_path

        model = AutoModelForCausalLM.from_pretrained(
            path_to_use,
            token=TOKEN,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir="/datasets/llms/",
            device_map="auto" 
        )
        return model

    def load_checkpoint(self):
        if os.path.exists(self.full_output_path):
            print(f"Found existing output file. Resuming...", flush=True)
            try:
                with open(self.full_output_path, "r") as f:
                    existing_data = json.load(f)
                
                # We expect the file to be smaller than the dataset now
                loaded_count = 0
                
                # Iterate through existing data and map to samples
                for i in range(len(existing_data)):
                    if i >= len(self.samples): 
                        break # Stop if file has more lines than dataset (unlikely)

                    data = existing_data[i]
                    sample = self.samples[i]

                    # Sanity Check
                    if data.get('original_sentence') != sample.original_sentence:
                        print(f"Mismatch at index {i}. Stopping load to prevent corruption.", flush=True)
                        break

                    # Load Scores
                    s_score = data.get('sentence_perplexity_score') or data.get('original_sentence_perplexity_score')
                    if s_score is not None:
                        sample.sentence_perplexity_score = s_score
                        sample.replaced_sentence_perplexity_score = data.get('replaced_sentence_perplexity_score')
                        loaded_count += 1
                            
                print(f"Successfully loaded {loaded_count} processed samples.", flush=True)
            except json.JSONDecodeError:
                print("Output file corrupted. Starting fresh.", flush=True)
        else:
            print("No existing output file found. Starting fresh.", flush=True)
    
    def perplexity_score(self, sentence):
        # Validation
        if not sentence or not isinstance(sentence, str) or len(sentence.strip()) == 0:
            return None

        # Tokenize (Truncated to prevent OOM)
        tokenize_input = self.tokenizer(
            sentence, 
            return_tensors='pt', 
            truncation=True, 
            max_length=4096, 
            return_attention_mask=True
        )
        
        input_ids = tokenize_input['input_ids'].to(self.model.device)
        attention_mask = tokenize_input['attention_mask'].to(self.model.device)
        
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"NaN loss detected for sentence len: {len(sentence)}", flush=True)
                # We raise error to see it, or you can return None to skip
                raise ValueError("Model returned NaN loss.")
                
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

                if i % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Calculate
                p_score = self.perplexity_score(sample.original_sentence)
                
                # CRITICAL: If score is None, we want to know why.
                # If it's None but sentence wasn't empty, something is wrong.
                if p_score is None and sample.original_sentence:
                     print(f"Warning: Score is None for index {i}", flush=True)

                sample.sentence_perplexity_score = p_score

                if hasattr(sample, 'replaced_sentence') and sample.replaced_sentence:
                    sample.replaced_sentence_perplexity_score = self.perplexity_score(sample.replaced_sentence)

                if (i + 1) % save_frequency == 0:
                    self.store_samples_as_json()
                    
        except KeyboardInterrupt:
            print("\nInterrupted. Saving...", flush=True)
            self.store_samples_as_json()
            raise
        except Exception as e:
            print(f"\nCRASH at index {i}: {e}", flush=True)
            self.store_samples_as_json()
            raise e

    def store_samples_as_json(self):
        """
        FIXED: Only saves samples that have been processed.
        This prevents 'null' entries from appearing at the end of the file.
        """
        output_data = []
        for sample in self.samples:
            
            # CHECK: If the main score is None, stop saving (or skip).
            # This assumes processing is sequential.
            if getattr(sample, 'sentence_perplexity_score', None) is None:
                # We break here because we haven't reached this part of the dataset yet.
                # This ensures the file doesn't fill up with future 'null' rows.
                break

            output_data.append({
                "original_sentence": sample.original_sentence,
                "replaced_sentence": sample.replaced_sentence,
                "original_pronoun": sample.original_pronoun,
                "replaced_pronoun": sample.replaced_pronoun,
                "gender": getattr(sample, 'gender', None),
                "sentence_perplexity_score": sample.sentence_perplexity_score,
                "replaced_sentence_perplexity_score": getattr(sample, 'replaced_sentence_perplexity_score', None)
            })
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        temp_file = self.full_output_path + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(output_data, f, indent=4)
        
        os.replace(temp_file, self.full_output_path)

    def run(self):
        self.add_perplexity_scores()
        # Final Save
        self.store_samples_as_json() 
        print(f"Output stored in {self.full_output_path}", flush=True)