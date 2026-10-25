from src.HousePrice.data import load_dataset, split_features_target
from src.HousePrice.predict import load_model, predict_batch

df = load_dataset()
X, y = split_features_target(df)

model = load_model()

batch = X.iloc[:10].copy()

results = predict_batch(batch, model)

print(results)
print("Actual prices:")
print(y.iloc[:10])