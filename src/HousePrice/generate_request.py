
import json

from HousePrice.data import load_dataset, split_features_target

df = load_dataset()
X, _ = split_features_target(df)

# Select a real house
sample_data = []
for i in range(2):
    sample = X.iloc[i].to_dict()
    sample_data.append(sample)

with open("sample_input.json", "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=4, default=str)

print("Generated sample_input.json")
print("Feature count:", len(sample_data))
