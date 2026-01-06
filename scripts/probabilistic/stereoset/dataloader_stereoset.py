import json
from tqdm import tqdm

class LMB(object):
    def __init__(self, location, json_obj=None):
        """
        Instantiates the LMB (StereoSet) object.

        Parameters
        ----------
        location (string): location of the json file.
        json_obj (dict): Optional direct injection of json object.
        """
        if json_obj is None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.samples = self.__create_samples__(self.json)

    def __create_samples__(self, json_data):
        created_samples = []
        
        # Navigate to the 'intrasentence' list based on the provided structure
        # Structure: root -> data -> intrasentence
        if 'data' in json_data and 'intrasentence' in json_data['data']:
            items = json_data['data']['intrasentence']
        else:
            # Fallback in case the list is passed directly
            items = json_data

        for item in items:
            # Extract parent metadata
            context = item.get('context')
            bias_type = item.get('bias_type')
            target = item.get('target')
            item_id = item.get('id')
            
            # extract specific sentences based on gold_label
            stereotype_sent = None
            anti_stereotype_sent = None
            unrelated_sent = None
            
            for sent_obj in item.get('sentences', []):
                gold_label = sent_obj.get('gold_label')
                text = sent_obj.get('sentence')
                
                if gold_label == 'stereotype':
                    stereotype_sent = text
                elif gold_label == 'anti-stereotype':
                    anti_stereotype_sent = text
                elif gold_label == 'unrelated':
                    unrelated_sent = text
            
            # Create the sample object containing all three variations
            created_samples.append(Sample(
                item_id=item_id,
                target=target,
                bias_type=bias_type,
                context=context,
                stereotype=stereotype_sent,
                anti_stereotype=anti_stereotype_sent,
                unrelated=unrelated_sent
            ))
            
        return created_samples
    
    def get_samples(self):
        return self.samples

class Sample(object):
    def __init__(self, item_id, target, bias_type, context, 
                 stereotype, anti_stereotype, unrelated):
        """ 
        Stores the data for a single context containing the triplet of sentences.
        """
        self.id = item_id
        self.target = target
        self.bias_type = bias_type
        self.context = context
        
        # The three variations
        self.stereotype_sentence = stereotype
        self.anti_stereotype_sentence = anti_stereotype
        self.unrelated_sentence = unrelated

    def __str__(self):
        return (f"Target: {self.target} ({self.bias_type})\n"
                f"Context: {self.context}\n"
                f"  [Stereotype]: {self.stereotype_sentence}\n"
                f"  [Anti-Stereo]: {self.anti_stereotype_sentence}\n"
                f"  [Unrelated]:   {self.unrelated_sentence}")