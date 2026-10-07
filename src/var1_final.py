import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

train = pd.read_csv("data/IMT2024066_train_var1.csv")
test = pd.read_csv("data/IMT2024066_test_var1.csv")
feats = [f"x{i}" for i in range(1, 7)]

model = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("scale", StandardScaler()),
    ("reg", Lasso(alpha=0.01, max_iter=100000)),
])
model.fit(train[feats].values, train["y"].values)

pred = model.predict(test[feats].values)

# sanity checks
train_pred = model.predict(train[feats].values)
print(f"Train MSE (all 1000 rows): {np.mean((train['y'] - train_pred) ** 2):.4f}")
print(f"Nonzero terms: {(model.named_steps['reg'].coef_ != 0).sum()}")
print(f"Train y range: [{train['y'].min():.2f}, {train['y'].max():.2f}]")
print(f"Pred  y range: [{pred.min():.2f}, {pred.max():.2f}]")
print(f"Train y mean/std: {train['y'].mean():.2f} / {train['y'].std():.2f}")
print(f"Pred  y mean/std: {pred.mean():.2f} / {pred.std():.2f}")

pd.DataFrame({"y": pred}).to_csv("predictions/IMT2024066_pred_var1.csv", index=False)
print("Saved predictions/IMT2024066_pred_var1.csv", pred.shape)