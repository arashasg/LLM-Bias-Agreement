import pandas as pd
import json
def validate_sample(row):
    if row["stereo_antistereo"] not in ["stereo", "antistereo"]:
        return False
    return True


path_to_file = "data/crows_pairs_anonymized.csv"

df = pd.read_csv(path_to_file)
results= []
print()
for _, row in df.iterrows():
    if not validate_sample(row):
        print(f"invalid line: {row}")
        continue
    if row["stereo_antistereo"] == "stereo":
        result = {
            "original_sentence": row["sent_more"],
            "replaced_sentence": row["sent_less"],
            "group": row["bias_type"]
        }
        results.append(result)
    else:
        result = {
            "original_sentence": row["sent_less"],
            "replaced_sentence": row["sent_more"],
            "group": row["bias_type"]
        }
        results.append(result)
# Write outputs
with open("crows-pairs.json", "w") as outfile_female:
    json.dump(results, outfile_female, indent=4)
