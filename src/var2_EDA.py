import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/IMT2024066_train_var2.csv")
test = pd.read_csv("data/IMT2024066_test_var2.csv")
feats = [c for c in test.columns]          # whatever the test file has
print("features:", feats)
print(train.shape, test.shape)
print(train.isna().sum().sum(), "missing in train,", test.isna().sum().sum(), "in test")

print("\nTRAIN summary:\n", train.describe().T)
print("\nTEST summary:\n", test.describe().T)

# range overlap: what fraction of test rows lie outside the train min/max?
lo, hi = train[feats].min(), train[feats].max()
outside = ((test[feats] < lo) | (test[feats] > hi))
print("\nFraction of test values outside train range, per feature:\n", outside.mean())
print("Fraction of test ROWS with any feature outside train range:",
      outside.any(axis=1).mean().round(3))

# clipping check, as in var1
print("\nClipped fraction train:\n", (train[feats].abs() == 1).mean())
print("Clipped fraction test:\n", (test[feats].abs() == 1).mean())

print("\nCorrelations:\n", train.corr().round(2))

# y vs each feature
fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
for ax, f in zip(np.atleast_1d(axes), feats):
    ax.scatter(train[f], train["y"], s=6, alpha=0.5)
    ax.set_xlabel(f); ax.set_ylabel("y")
plt.tight_layout()
plt.savefig("figures/var2_scatter.png", dpi=150)
plt.show()

# train vs test feature distributions
fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
for ax, f in zip(np.atleast_1d(axes), feats):
    ax.hist(train[f], bins=30, alpha=0.5, label="train", density=True)
    ax.hist(test[f], bins=30, alpha=0.5, label="test", density=True)
    ax.set_title(f); ax.legend()
plt.tight_layout()
plt.savefig("figures/var2_hist.png", dpi=150)
plt.show()