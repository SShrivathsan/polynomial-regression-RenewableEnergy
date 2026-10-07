import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

train = pd.read_csv("data/IMT2024066_train_var1.csv")
feats = [f"x{i}" for i in range(1, 7)]
X, y = train[feats].values, train["y"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

DEGREES = [4, 5, 6]

pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("reg", ElasticNet(max_iter=100000)),
])
grid = {
    "poly__degree": DEGREES,
    "reg__l1_ratio": [0.3, 0.5, 0.7, 0.9, 1.0],
    "reg__alpha": [0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
}
gs = GridSearchCV(pipe, grid, cv=kf, scoring="neg_mean_squared_error",
                  return_train_score=True, n_jobs=-1)
gs.fit(X, y)

r = pd.DataFrame(gs.cv_results_)
r["val_mse"] = -r["mean_test_score"]
r["train_mse"] = -r["mean_train_score"]
r["val_std"] = r["std_test_score"]
r = r.rename(columns={"param_poly__degree": "degree",
                      "param_reg__l1_ratio": "l1_ratio",
                      "param_reg__alpha": "alpha"})
cols = ["degree", "l1_ratio", "alpha", "train_mse", "val_mse", "val_std"]
r[cols].to_csv("figures/var1_enet_grid.csv", index=False)

print("Best alpha per (degree, l1_ratio):")
best = r.loc[r.groupby(["degree", "l1_ratio"])["val_mse"].idxmin()]
print(best[cols].round(4).to_string(index=False))

edge = (np.abs(X) == 1).sum(axis=1) >= 3
pred = cross_val_predict(gs.best_estimator_, X, y, cv=kf)
print(f"\nOverall best: {gs.best_params_}")
print(f"overall MSE={np.mean((y-pred)**2):.4f}, "
      f"edge MSE={np.mean((y[edge]-pred[edge])**2):.4f}, "
      f"interior MSE={np.mean((y[~edge]-pred[~edge])**2):.4f}")
coef = gs.best_estimator_.named_steps["reg"].coef_
print(f"Nonzero terms: {(coef != 0).sum()} of {len(coef)}")