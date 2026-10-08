# Polynomial Regression: Geothermal Plant Assignment

Roll number: IMT2024066

Two regression problems solved with polynomial regression only:
- **var1** (turbine power score, 6 features): Lasso, degree 5, alpha = 0.01
- **var2** (thermal anomaly score, 3 features): Ridge, degree 10, alpha = 1

The full write-up is in `report/IMT2024066_report.pdf`.

## Setup
Place the four data files in `data/` (`IMT2024066_train_var1.csv`, `IMT2024066_test_var1.csv`, `IMT2024066_train_var2.csv`, `IMT2024066_test_var2.csv`). Run all scripts from the repository root.

## Structure
```
data/          input CSVs
src/           all code (var1_*.py and var2_*.py)
figures/       plots and CSV outputs of the model searches
predictions/   final prediction files
report/        PDF report
```

## Reproducing the results

Scripts are numbered in the order they were run. Each reads from `data/` and writes to `figures/` where relevant.

**Problem 1 (var1)**
1. `python src/var1_EDA.py`: data checks and plots
2. `python src/var1_sweep.py`: degree sweep over feature subsets (unregularized)
3. `python src/var1_unreg.py`: unregularized sweep on all six features
4. `python src/var1_reg.py`: Ridge and Lasso, degree x alpha grid
5. `python src/var1_lasso_deg.py`: Lasso at degrees 5 to 7
6. `python src/var1_enet.py`: ElasticNet grid
7. `python src/var1_final.py`: final fit and predictions

**Problem 2 (var2)**
1. `python src/var2_EDA.py`
2. `python src/var2_sweep.py`: all feature subsets, unregularized
3. `python src/var2_reg.py`: Ridge and Lasso grid
4. `python src/var2_enets.py`: ElasticNet grid
5. `python src/var2_final.py`: final fit and predictions

Only the two `*_final.py` scripts produce the submission files: `predictions/IMT2024066_pred_var1.csv` and `predictions/IMT2024066_pred_var2.csv`. All other scripts are the model-selection experiments described in the report.

## Method in one line
`PolynomialFeatures -> StandardScaler -> Ridge/Lasso/ElasticNet`, with degree and regularization chosen by cross-validation.