import json
import random
from tqdm import tqdm

class LMB(object):
    def __init__(self, location, json_obj=None, seed=42):
        """
        Instantiates the StereoSet object.

        Parameters
        ----------
        location (string): location of the StereoSet.json file.
        json_obj (dict, optional): Direct JSON object if file is not used.
        seed (int): Random seed for reproducibility of the shuffle.
        """
        self.seed = None

        if json_obj is None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.samples = self.__create_samples__(self.json)

    def __create_samples__(self, samples):
        created_samples = []
        for sample in samples:
            created_samples.append(Sample(
                sample['sentence'], 
                sample['axis'], 
                sample['descriptor'], 
                sample['gender']
            ))
        
        # --- SHUFFLE LOGIC ---
        # We shuffle the list here to ensure axes and descriptors are mixed.
        random.seed(self.seed)
        random.shuffle(created_samples)

        # Move 'ability' axis to the end of the list
        non_ability_samples = [s for s in created_samples if s.axis != 'ability']
        ability_samples = [s for s in created_samples if s.axis == 'ability']
        
        created_samples = non_ability_samples + ability_samples
        
        return created_samples
    
    def get_samples(self):
        return self.samples

class Sample(object):
    def __init__(self, original_sentence, axis, descriptor, gender):
        self.sentence = original_sentence
        self.axis = axis
        self.descriptor = descriptor
        self.gender = gender

    def convert_to_list(self, input_string):
        sanitized_string = input_string.replace("'", "").strip()
        return [item.strip() for item in sanitized_string.split(',')] if ',' in sanitized_string else [sanitized_string]

    def __str__(self):
        # Fixed bug: changed self.original_sentence to self.sentence to match __init__
        s = f"original sentence: {self.sentence}" 
        return s