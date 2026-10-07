import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold, GridSearchCV

warnings.filterwarnings("ignore")   # silence ConvergenceWarning noise
pd.set_option("display.width", 200)
train = pd.read_csv("data/IMT2024066_train_var2.csv")
feats = ["x1", "x2", "x3"]
X, y = train[feats].values, train["y"].values
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # same splits as var2_sweep / var2_reg

pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("reg", ElasticNet(max_iter=20000, tol=1e-3)),
])
grid = {
    "poly__degree": [8, 10],
    "reg__l1_ratio": [0.1, 0.3, 0.5],
    "reg__alpha": [0.0005, 0.001, 0.002, 0.004, 0.008],
}
gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error",
                  return_train_score=True, n_jobs=-1, verbose=1)
gs.fit(X, y)

r = pd.DataFrame(gs.cv_results_)
r["val_mse"] = -r["mean_test_score"]
r["train_mse"] = -r["mean_train_score"]
r["val_std"] = r["std_test_score"]
r = r.rename(columns={"param_poly__degree": "degree",
                      "param_reg__l1_ratio": "l1_ratio",
                      "param_reg__alpha": "alpha"})
cols = ["degree", "l1_ratio", "alpha", "train_mse", "val_mse", "val_std"]
r[cols].to_csv("figures/var2_enet_grid.csv", index=False)

print("Best alpha per (degree, l1_ratio):")
best = r.loc[r.groupby(["degree", "l1_ratio"])["val_mse"].idxmin()]
print(best[cols].round(4).to_string(index=False))

print(f"\nOverall best: {gs.best_params_}, val MSE = {-gs.best_score_:.4f}")
coef = gs.best_estimator_.named_steps["reg"].coef_
print(f"Nonzero terms: {(coef != 0).sum()} of {len(coef)}")

bd = gs.best_params_["poly__degree"]
piv = r[r["degree"] == bd].pivot(index="l1_ratio", columns="alpha", values="val_mse")
piv.columns = [f"{a:g}" for a in piv.columns]
print(f"\nVal MSE at degree {bd} (rows: l1_ratio, columns: alpha):")
print(piv.round(3).to_string())