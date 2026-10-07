import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RepeatedKFold, GridSearchCV

pd.set_option("display.width", 200)
train = pd.read_csv("data/IMT2024066_train_var2.csv")
feats = ["x1", "x2", "x3"]
X, y = train[feats].values, train["y"].values
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # same splits as var2_sweep

def run(name, reg, degrees, alphas):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", reg),
    ])
    gs = GridSearchCV(pipe, {"poly__degree": degrees, "reg__alpha": alphas},
                      cv=cv, scoring="neg_mean_squared_error",
                      return_train_score=True, n_jobs=-1)
    gs.fit(X, y)
    r = pd.DataFrame(gs.cv_results_)
    r["val_mse"] = -r["mean_test_score"]
    r["train_mse"] = -r["mean_train_score"]
    r["val_std"] = r["std_test_score"]
    r = r.rename(columns={"param_poly__degree": "degree", "param_reg__alpha": "alpha"})
    r["model"] = name

    print(f"\n{name}: best alpha per degree")
    best = r.loc[r.groupby("degree")["val_mse"].idxmin()]
    print(best[["degree", "alpha", "train_mse", "val_mse", "val_std"]].round(4).to_string(index=False))

    print(f"\n{name}: validation MSE by degree (rows) and alpha (columns)")
    piv = r.pivot(index="degree", columns="alpha", values="val_mse")
    piv.columns = [f"{a:g}" for a in piv.columns]
    print(piv.round(3).to_string())
    return gs, r

ridge_gs, ridge_r = run("Ridge", Ridge(), [6, 8, 10, 12, 14],
                        list(np.logspace(-4, 2, 13)))
lasso_gs, lasso_r = run("Lasso", Lasso(max_iter=100000), [6, 8, 10, 12],
                        [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])

pd.concat([ridge_r, lasso_r])[["model", "degree", "alpha", "train_mse", "val_mse", "val_std"]] \
  .to_csv("figures/var2_reg_grid.csv", index=False)

for name, gs in [("Ridge", ridge_gs), ("Lasso", lasso_gs)]:
    print(f"\n{name} overall best: {gs.best_params_}, val MSE = {-gs.best_score_:.4f}")
coef = lasso_gs.best_estimator_.named_steps["reg"].coef_
print(f"Lasso nonzero terms: {(coef != 0).sum()} of {len(coef)}")