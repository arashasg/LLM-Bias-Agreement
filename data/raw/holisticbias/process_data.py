import pandas as pd
import numpy as np
import json
from tqdm import tqdm

path = "sentences.csv"
output_path = "holistic_bias.json"
results= []
df = pd.read_csv(path)
print(np.unique(df["descriptor_preference"]))
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    json_data = {
        "sentence": row["text"],
        "axis": row["axis"],
        "descriptor": row["descriptor"],
        "gender": row["noun_gender"],
    }
    results.append(json_data)

with open(output_path, "w") as outfile:
    json.dump(results, outfile, indent=4)