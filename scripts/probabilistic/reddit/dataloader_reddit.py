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
        # Iterate through the list of dictionaries in the JSON
        for sample in samples:
            created_samples.append(Sample(
                original_words=sample['original_words'],
                replaced_words=sample['replaced_words'],
                original_sentence=sample['original_sentence'],
                replaced_sentence=sample['replaced_sentence'],
                group=sample['group']
            ))
        return created_samples

    def get_samples(self):
        return self.samples


class Sample(object):
    def __init__(self, original_words, replaced_words, original_sentence, 
                 replaced_sentence, group):
        """ 
        Stores the data for a single entry from the JSON.
        """
        self.original_words = original_words
        self.replaced_words = replaced_words
        self.original_sentence = original_sentence
        self.replaced_sentence = replaced_sentence
        self.group = group

    def __str__(self):
        s = (f"Group: {self.group}\n"
             f"Original Words: {self.original_words} -> Replaced: {self.replaced_words}\n"
             f"Original Sent: {self.original_sentence}\n"
             f"Replaced Sent: {self.replaced_sentence}")
        return s