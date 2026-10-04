import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

train = pd.read_csv("data/IMT2024066_train_var1.csv")
y = train["y"].values

subsets = {
    "x1-x3":    (["x1", "x2", "x3"], 10),
    "x3,x5,x6": (["x3", "x5", "x6"], 10),
    "all six":  ([f"x{i}" for i in range(1, 7)], 5),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, (cols, max_deg) in subsets.items():
    X = train[cols].values
    for d in range(1, max_deg + 1):
        model = make_pipeline(
            PolynomialFeatures(degree=d, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        cv = cross_validate(model, X, y, cv=kf,
                            scoring="neg_mean_squared_error",
                            return_train_score=True)
        rows.append({
            "subset": name, "degree": d,
            "train_mse": -cv["train_score"].mean(),
            "val_mse":   -cv["test_score"].mean(),
        })

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

# plot train vs validation MSE per subset
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, g) in zip(axes, res.groupby("subset", sort=False)):
    ax.plot(g["degree"], g["train_mse"], "o-", label="train")
    ax.plot(g["degree"], g["val_mse"], "o-", label="validation")
    ax.set_title(name); ax.set_xlabel("degree"); ax.set_ylabel("MSE")
    ax.set_yscale("log"); ax.legend()
plt.tight_layout()
plt.savefig("figures/var1_degree_sweep.png", dpi=150)
plt.show()