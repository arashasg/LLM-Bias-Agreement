import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from resources import TOKEN
import os
import json
import dataloader_stereoset as dataloader
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
            device_map="auto" 
        )
        return model
    
    def load_checkpoint(self):
        if os.path.exists(self.full_output_path):
            print(f"Found existing output file at {self.full_output_path}. Resuming...", flush=True)
            try:
                with open(self.full_output_path, "r") as f:
                    existing_data = json.load(f)
                
                if len(existing_data) > len(self.samples):
                    existing_data = existing_data[:len(self.samples)]
                
                loaded_count = 0
                for i in range(len(existing_data)):
                    data = existing_data[i]
                    sample = self.samples[i]

                    # Sanity Check: StereoSet has unique IDs
                    if data.get('id') != sample.id:
                        print(f"Mismatch at index {i} (ID: {data.get('id')} vs {sample.id}). Stopping load.", flush=True)
                        break

                    s_score = data.get('stereotype_perplexity_score')
                    
                    if s_score is not None:
                        sample.stereotype_perplexity_score = s_score
                        sample.anti_stereotype_perplexity_score = data.get('anti_stereotype_perplexity_score')
                        sample.unrelated_perplexity_score = data.get('unrelated_perplexity_score')
                        loaded_count += 1
                            
                print(f"Successfully loaded {loaded_count} processed samples.", flush=True)
            except json.JSONDecodeError:
                print("Output file exists but is corrupted. Starting from scratch.", flush=True)
        else:
            print("No existing output file found. Starting fresh.", flush=True)

    def perplexity_score(self, sentence):
        # 1. Validation
        if not sentence or not isinstance(sentence, str) or len(sentence.strip()) == 0:
            return None

        # 2. Tokenize (Truncated)
        tokenize_input = self.tokenizer(
            sentence, 
            return_tensors='pt',
            truncation=True,
            max_length=4096, # Prevent OOM on outliers
            return_attention_mask=True
        )
        
        input_ids = tokenize_input['input_ids'].to(self.model.device)
        attention_mask = tokenize_input['attention_mask'].to(self.model.device)
        
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"!!! NaN LOSS DETECTED !!! Sentence: {sentence[:50]}...", flush=True)
                raise ValueError("Model returned NaN loss.")
            
            perplexity = math.exp(loss.item())
        
        del outputs
        del input_ids
        del attention_mask
        return perplexity

    def add_perplexity_scores(self):
        save_frequency = 10 
        
        try:
            for i, sample in enumerate(tqdm(self.samples, desc="Processing StereoSet samples", unit="sample")):
                
                # Skip if already calculated
                if getattr(sample, 'stereotype_perplexity_score', None) is not None:
                    continue

                # Periodic Garbage Collection
                if i % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Calculate Scores (Fail Loudly)
                # Stereotype
                p_stereo = self.perplexity_score(sample.stereotype_sentence)
                if p_stereo is None and sample.stereotype_sentence:
                     print(f"Warning: Score is None for stereotype at index {i}", flush=True)
                sample.stereotype_perplexity_score = p_stereo

                # Anti-Stereotype
                p_anti = self.perplexity_score(sample.anti_stereotype_sentence)
                if p_anti is None and sample.anti_stereotype_sentence:
                     print(f"Warning: Score is None for anti-stereotype at index {i}", flush=True)
                sample.anti_stereotype_perplexity_score = p_anti

                # Unrelated
                p_unrel = self.perplexity_score(sample.unrelated_sentence)
                if p_unrel is None and sample.unrelated_sentence:
                     print(f"Warning: Score is None for unrelated at index {i}", flush=True)
                sample.unrelated_perplexity_score = p_unrel

                # Save
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
        FIXED: Only saves samples that have been processed to avoid 'null' tail.
        """
        output_data = []
        for sample in self.samples:
            
            # STOP saving if we hit a sample that hasn't been processed yet.
            if getattr(sample, 'stereotype_perplexity_score', None) is None:
                break

            output_data.append({
                # Metadata
                "id": sample.id,
                "target": sample.target,
                "bias_type": sample.bias_type,
                "context": sample.context,
                
                # Sentences
                "stereotype_sentence": sample.stereotype_sentence,
                "anti_stereotype_sentence": sample.anti_stereotype_sentence,
                "unrelated_sentence": sample.unrelated_sentence,
                
                # Calculated Scores
                "stereotype_perplexity_score": getattr(sample, 'stereotype_perplexity_score', None),
                "anti_stereotype_perplexity_score": getattr(sample, 'anti_stereotype_perplexity_score', None),
                "unrelated_perplexity_score": getattr(sample, 'unrelated_perplexity_score', None)
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
        self.store_samples_as_json() # Final save
        print(f"Output stored in {self.full_output_path}", flush=True)