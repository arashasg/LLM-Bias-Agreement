import json
import re
import os
import nltk
from nltk.tokenize import word_tokenize

# Ensure that NLTK is properly downloaded for tokenization
nltk.download('punkt')

def remove_brackets(sentence):
    return re.sub(r'\[|\]', '', sentence).strip()

def remove_determiner(occupation):
    # Tokenize the input string
    tokens = word_tokenize(occupation)
    
    # Define a set of determiners
    determiners = {"the", "a", "an"}
    
    # Remove determiners and join the remaining tokens
    filtered_tokens = [token for token in tokens if token.lower() not in determiners]
    return " ".join(filtered_tokens)

def load_occupations(file_path):
    with open(file_path, "r") as file:
        return set(line.strip().lower() for line in file)

def process_sentences(file1_path, file2_path, female_occupations_path, male_occupations_path, output_path, output_female_path, output_male_path):
    # Load occupations
    female_occupations = load_occupations(female_occupations_path)
    male_occupations = load_occupations(male_occupations_path)

    # Read input files
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
        sentences1 = file1.readlines()
        sentences2 = file2.readlines()

    results_female = []
    results_male = []
    results = []

    for line1, line2 in zip(sentences1, sentences2):
        id1, sentence1 = line1.split(" ", 1)
        id2, sentence2 = line2.split(" ", 1)

        if id1 != id2:
            raise ValueError(f"Mismatched IDs: {id1} and {id2}")

        # Extract words inside brackets
        brackets1 = re.findall(r'\[(.*?)\]', sentence1)
        brackets2 = re.findall(r'\[(.*?)\]', sentence2)

        if not brackets1 or not brackets2:
            raise ValueError(f"No bracketed words found in sentences: {sentence1} | {sentence2}")

        # Initialize variables to hold pronouns and occupations
        pronoun1, pronoun2 = None, None
        occupation1, occupation2 = None, None

        # Check each bracketed phrase for pronouns and occupations
        for phrase in brackets1:
            tokens = word_tokenize(phrase)
            if any(token.lower() in {"she", "her", "he", "him", "his"} for token in tokens):
                pronoun1 = next(token for token in tokens if token.lower() in {"she", "her", "he", "him", "his"})
            # Check if the entire phrase is in the occupation sets
            cleaned_phrase = phrase.lower().strip()
            if remove_determiner(cleaned_phrase) in female_occupations.union(male_occupations):
                occupation1 = cleaned_phrase

        for phrase in brackets2:
            tokens = word_tokenize(phrase)
            if any(token.lower() in {"she", "her", "he", "him", "his"} for token in tokens):
                pronoun2 = next(token for token in tokens if token.lower() in {"she", "her", "he", "him", "his"})
            # Check if the entire phrase is in the occupation sets
            cleaned_phrase = phrase.lower().strip()
            if remove_determiner(cleaned_phrase) in female_occupations.union(male_occupations):
                occupation2 = cleaned_phrase

        if not occupation1 or not occupation2 or not pronoun1 or not pronoun2:
            raise ValueError(f"Missing expected bracketed words in sentences: {id1} | {id2} | {sentence1} | {sentence2} | {occupation2} | {occupation2} | {pronoun1} | {pronoun2}")

        if remove_determiner(occupation1.lower()) != remove_determiner(occupation2.lower()):
            print(f"Mismatched occupations in sentences: {id1} | {id2} | {sentence1} | {sentence2} | {occupation2} | {occupation2} | {pronoun1} | {pronoun2}")
            continue

        group = "gender"
        
        result = {
            "original_words": pronoun1 if pronoun1.lower() in {"she", "her"} else pronoun2,
            "replaced_words": pronoun2 if pronoun2.lower() in {"he", "him", "his"} else pronoun1,
            "original_sentence": remove_brackets(sentence1.strip() if pronoun1.lower() in {"she", "her"} else sentence2.strip()),
            "replaced_sentence": remove_brackets(sentence2.strip() if pronoun2.lower() in {"he", "him", "his"} else sentence1.strip()),
            "group": group,
            "occupation" : remove_determiner(occupation1.lower().strip()),
        }

        # Check and store results based on occupation
        if occupation1:
            cleaned_occupation = occupation1.lower().strip()
            
            if remove_determiner(cleaned_occupation) in female_occupations:
                results_female.append(result)
                result["occupation_gender"] = "female"
            elif remove_determiner(cleaned_occupation) in male_occupations:
                results_male.append(result)
                result["occupation_gender"] = "male"
            results.append(result)

    # Write outputs
    with open(output_female_path, "w") as outfile_female:
        json.dump(results_female, outfile_female, indent=4)

    with open(output_male_path, "w") as outfile_male:
        json.dump(results_male, outfile_male, indent=4)

    with open(output_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

base = "./data/winobiasp/data/"
types = ["type1", "type2"]
files = ["dev", "test"]

for t in types:
    for f in files:
        anti_stereotyped_file = os.path.join(base, f"anti_stereotyped_{t}.txt.{f}")
        pro_stereotyped_file = os.path.join(base, f"pro_stereotyped_{t}.txt.{f}")
        output_female_path = os.path.join(base, f"output_female_{t}_{f}.json")
        output_male_path = os.path.join(base, f"output_male_{t}_{f}.json")
        output_path = os.path.join(base, f"output_with_occupation_{t}_{f}.json")
        print(f"processing...  type: {t}, file: {f}")

        # Call the process_sentences function
        process_sentences(
            anti_stereotyped_file, pro_stereotyped_file,
            os.path.join(base, "female_occupations.txt"), os.path.join(base, "male_occupations.txt"), output_path,
            output_female_path, output_male_path
        )