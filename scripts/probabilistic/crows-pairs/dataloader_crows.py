import json
from tqdm import tqdm

class LMB(object):
    def __init__(self, location, json_obj=None):
        """
        Instantiates the LMB object.

        Parameters
        ----------
        location (string): location of the json file.
        json_obj (list): Optional direct injection of json object.
        """
        if json_obj is None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.samples = self.__create_samples__(self.json)

    def __create_samples__(self, samples):
        created_samples = []
        for sample in samples:
            # Updated to extract only the keys present in your new structure
            created_samples.append(Sample(
                original_sentence=sample['original_sentence'],
                replaced_sentence=sample['replaced_sentence'],
                group=sample['group']
            ))
        return created_samples
    
    def get_samples(self):
        return self.samples

class Sample(object):
    def __init__(self, original_sentence, replaced_sentence, group):
        """ 
        Stores the data for a single entry.
        """
        self.original_sentence = original_sentence
        self.replaced_sentence = replaced_sentence
        self.group = group

    def __str__(self):
        return (f"Group: {self.group}\n"
                f"Original: {self.original_sentence}\n"
                f"Replaced: {self.replaced_sentence}")