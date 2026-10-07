import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

train = pd.read_csv("data/IMT2024066_train_var1.csv")
feats = [f"x{i}" for i in range(1, 7)]
X, y = train[feats].values, train["y"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def run(name, reg, degrees, alphas):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", reg),
    ])
    grid = {"poly__degree": degrees, "reg__alpha": alphas}
    gs = GridSearchCV(pipe, grid, cv=kf, scoring="neg_mean_squared_error",
                      return_train_score=True, n_jobs=-1)
    gs.fit(X, y)
    r = pd.DataFrame(gs.cv_results_)
    r["val_mse"] = -r["mean_test_score"]
    r["train_mse"] = -r["mean_train_score"]
    r["val_std"] = r["std_test_score"]
    r["model"] = name
    r = r.rename(columns={"param_poly__degree": "degree", "param_reg__alpha": "alpha"})
    print(f"\n{name}: best degree/alpha per degree")
    best = r.loc[r.groupby("degree")["val_mse"].idxmin()]
    print(best[["degree", "alpha", "train_mse", "val_mse", "val_std"]].round(4).to_string(index=False))
    return gs, r

ridge_gs, ridge_r = run("Ridge", Ridge(), [3, 4, 5, 6], list(np.logspace(-3, 3, 13)))
lasso_gs, lasso_r = run("Lasso", Lasso(max_iter=50000), [3, 4, 5], list(np.logspace(-4, -0.5, 8)))

pd.concat([ridge_r, lasso_r])[["model", "degree", "alpha", "train_mse", "val_mse"]] \
  .to_csv("figures/var1_reg_grid.csv", index=False)

# Edge-heavy check: how do the winners do on rows with >=3 features at +/-1?
# (test has ~50% clipped per feature, train only ~30%)
edge = (np.abs(X) == 1).sum(axis=1) >= 3
print(f"\nEdge rows in train: {edge.sum()} of {len(X)}")
for name, gs in [("Ridge", ridge_gs), ("Lasso", lasso_gs)]:
    pred = cross_val_predict(gs.best_estimator_, X, y, cv=kf)
    print(f"{name} best {gs.best_params_}: "
          f"overall MSE={np.mean((y-pred)**2):.4f}, "
          f"edge MSE={np.mean((y[edge]-pred[edge])**2):.4f}, "
          f"interior MSE={np.mean((y[~edge]-pred[~edge])**2):.4f}")

# Lasso sparsity: how many terms survive?
best = lasso_gs.best_estimator_
names = best.named_steps["poly"].get_feature_names_out(feats)
coef = best.named_steps["reg"].coef_
keep = np.argsort(-np.abs(coef))[:15]
print(f"\nLasso nonzero terms: {(coef != 0).sum()} of {len(coef)}")
for i in keep:
    if coef[i] != 0:
        print(f"  {names[i]:<14} {coef[i]: .3f}")