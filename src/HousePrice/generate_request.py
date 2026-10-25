
import json

from HousePrice.data import load_dataset, split_features_target

df = load_dataset()
X, _ = split_features_target(df)

# Select a real house
sample = X.iloc[0].to_dict()

with open("sample_input.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=4, default=str)

print("Generated sample_input.json")
print("Feature count:", len(sample))
