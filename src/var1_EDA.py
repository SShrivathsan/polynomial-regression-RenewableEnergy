import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/IMT2024066_train_var1.csv")
test  = pd.read_csv("data/IMT2024066_test_var1.csv")
feats = [f"x{i}" for i in range(1, 7)]

# 1. basic checks
print(train.shape, test.shape)
print(train.describe().T)
print("Clipped fraction train:\n", (train[feats].abs() == 1).mean())
print("Clipped fraction test:\n",  (test[feats].abs() == 1).mean())

# 2. y vs each feature
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, f in zip(axes.ravel(), feats):
    ax.scatter(train[f], train["y"], s=6, alpha=0.5)
    ax.set_xlabel(f); ax.set_ylabel("y")
plt.tight_layout(); plt.show()

# 3. train vs test feature distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, f in zip(axes.ravel(), feats):
    ax.hist(train[f], bins=30, alpha=0.5, label="train", density=True)
    ax.hist(test[f],  bins=30, alpha=0.5, label="test",  density=True)
    ax.set_title(f); ax.legend()
plt.tight_layout(); plt.show()

# 4. correlation heatmap
print(train.corr().round(2))