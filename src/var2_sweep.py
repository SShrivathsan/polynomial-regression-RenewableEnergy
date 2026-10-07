import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate

train = pd.read_csv("data/IMT2024066_train_var2.csv")
y = train["y"].values
feats = ["x1", "x2", "x3"]
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # 15 folds, steadier with heavy tails

rows = []
for k in (1, 2, 3):
    max_deg = 20 if k < 3 else 12
    for cols in combinations(feats, k):
        X = train[list(cols)].values
        name = "+".join(cols)
        for d in range(1, max_deg + 1):
            model = make_pipeline(
                PolynomialFeatures(degree=d, include_bias=False),
                StandardScaler(),
                LinearRegression(),
            )
            r = cross_validate(
                model, X, y, cv=cv,
                scoring=["neg_mean_squared_error", "r2"],
                return_train_score=True,
            )
            rows.append({
                "subset": name,
                "degree": d,
                "train_mse": -r["train_neg_mean_squared_error"].mean(),
                "val_mse": -r["test_neg_mean_squared_error"].mean(),
                "val_std": r["test_neg_mean_squared_error"].std(),
                "val_r2": r["test_r2"].mean(),
            })
        print("done", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv("figures/var2_unregularized.csv", index=False)

pd.set_option("display.width", 200)
print("\nValidation MSE by degree (rows) and feature subset (columns):")
print(res.pivot(index="degree", columns="subset", values="val_mse").round(2).to_string())

print("\nBest degree per subset:")
best = res.loc[res.groupby("subset")["val_mse"].idxmin()]
print(best[["subset", "degree", "train_mse", "val_mse", "val_std", "val_r2"]]
      .sort_values("val_mse").round(3).to_string(index=False))