import numpy as np
import pandas as pd
from math import comb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

train = pd.read_csv("data/IMT2024066_train_var1.csv")
feats = [f"x{i}" for i in range(1, 7)]
X, y = train[feats].values, train["y"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for d in range(1, 7):
    model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    cv = cross_validate(
        model, X, y, cv=kf,
        scoring=["neg_mean_squared_error", "r2"],
        return_train_score=True,
    )
    rows.append({
        "degree": d,
        "n_terms": comb(6 + d, d) - 1,
        "train_mse": -cv["train_neg_mean_squared_error"].mean(),
        "val_mse": -cv["test_neg_mean_squared_error"].mean(),
        "val_std": cv["test_neg_mean_squared_error"].std(),
        "val_r2": cv["test_r2"].mean(),
    })

res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.to_csv("figures/var1_unregularized.csv", index=False)