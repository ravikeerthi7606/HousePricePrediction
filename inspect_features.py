
from src.HousePrice.data import load_dataset, split_features_target

df = load_dataset()
X, y = split_features_target(df)

print("Feature count:", X.shape[1])
print("\nFeature types:\n")
print(X.dtypes.to_string())

print("\nNumeric features:", len(X.select_dtypes(include="number").columns))
print(
    "Categorical features:",
    len(X.select_dtypes(include=["object", "category", "string", "bool"]).columns)
)

print("\nMissing values:")
print(X.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False))
