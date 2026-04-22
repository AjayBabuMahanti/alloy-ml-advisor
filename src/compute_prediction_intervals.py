"""
Compute split-conformal prediction intervals for the regressor.

Methodology:
  We use 5-fold GroupKFold cross-validation on the training data as a calibration
  set. For each fold, we hold out one group of alloys, fit the regressor on the
  rest, and record the signed residuals on the held-out fold. Pooling the
  residuals across all folds gives an empirical calibration set.

  For a target coverage level alpha (e.g. 0.9), the prediction interval for a
  new point x is:

     [ y_hat(x) - q_alpha ,  y_hat(x) + q_alpha ]

  where q_alpha is the alpha-quantile of |residuals|. This is the standard
  split-conformal approach (Angelopoulos & Bates, 2021), valid under the
  exchangeability assumption.

  We save the quantiles per target and per alpha to models/prediction_intervals.joblib
  so the Streamlit app can display intervals without retraining.
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone as sk_clone
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "alloy_dataset_final.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TARGETS = ["UTS_MPa", "YS_MPa", "Elongation_pct", "Hardness_HB", "Modulus_GPa"]
NUMERIC_COLS = [
    "Density_g_cm3", "Al_wt", "Cu_wt", "Fe_wt", "Mg_wt", "Mn_wt", "Si_wt",
    "Ti_wt", "Zn_wt", "Cr_wt", "Others_wt",
    "Mg_Si_ratio", "Zn_Mg_sum", "Cu_Mg_sum", "total_solute",
]
REG_CAT_COLS = ["Series", "Temper", "Form", "Processing"]
ALPHAS = [0.80, 0.90, 0.95]   # coverage levels we will support in the UI

print("=" * 60)
print("Computing conformal prediction intervals")
print("=" * 60)

# --- load + prepare data ---------------------------------------------------
df = pd.read_csv(DATA_PATH)
comp_cols = ["Al_wt", "Cu_wt", "Fe_wt", "Mg_wt", "Mn_wt",
             "Si_wt", "Ti_wt", "Zn_wt", "Cr_wt", "Others_wt"]
comp_sum = df[comp_cols].sum(axis=1)
df["Al_wt"] = df["Al_wt"] - (comp_sum - 100.0)
df["Al_wt"] = df["Al_wt"].clip(lower=70.0)

# Derived features (match training script)
df["Mg_Si_ratio"] = df["Mg_wt"] / (df["Si_wt"] + 0.01)
df["Zn_Mg_sum"] = df["Zn_wt"] + df["Mg_wt"]
df["Cu_Mg_sum"] = df["Cu_wt"] + df["Mg_wt"]
df["total_solute"] = df[["Cu_wt", "Mg_wt", "Mn_wt", "Si_wt", "Zn_wt", "Cr_wt"]].sum(axis=1)

# Exclude die-cast (matches main training script's scope)
df = df[df["Processing"] != "Die Cast"].reset_index(drop=True)

X = df[NUMERIC_COLS + REG_CAT_COLS].copy()
y = df[TARGETS].copy()
groups = df["Alloy"].str.replace(r"_v\d+$", "", regex=True)

print(f"Calibration data: {len(df)} records across {groups.nunique()} groups")

# --- load the trained pipeline for cloning ---------------------------------
trained_reg = joblib.load(os.path.join(MODELS_DIR, "regressor_lgbm_multi_pipeline.joblib"))
print(f"Loaded trained regressor: {type(trained_reg).__name__}")

# --- 5-fold GroupKFold: collect signed residuals per fold ------------------
print("\nRunning 5-fold GroupKFold to collect residuals...")
gkf = GroupKFold(n_splits=5)
residuals = {t: [] for t in TARGETS}

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    pipe = sk_clone(trained_reg)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGETS, index=X_val.index)

    for t in TARGETS:
        mask = ~y_val[t].isna()
        if mask.sum() == 0:
            continue
        residuals[t].extend((y_val.loc[mask, t].values - y_pred_df.loc[mask, t].values).tolist())
    print(f"  fold {fold}: val size={len(val_idx)}")

# --- compute absolute-residual quantiles per alpha -------------------------
print("\nCalibration quantiles |residual| per target and coverage level:")
intervals = {"alphas": ALPHAS, "targets": TARGETS, "quantiles": {}}
for t in TARGETS:
    abs_res = np.abs(np.array(residuals[t]))
    q_for_t = {}
    for alpha in ALPHAS:
        q = float(np.quantile(abs_res, alpha))
        q_for_t[alpha] = q
    intervals["quantiles"][t] = q_for_t
    mean_res = abs_res.mean()
    print(f"  {t:18s} | n={len(abs_res):4d} | mean|r|={mean_res:6.2f} | "
          f"q80={q_for_t[0.80]:6.2f} | q90={q_for_t[0.90]:6.2f} | q95={q_for_t[0.95]:6.2f}")

# --- save ------------------------------------------------------------------
out_path = os.path.join(MODELS_DIR, "prediction_intervals.joblib")
joblib.dump(intervals, out_path)
print(f"\nSaved: {out_path}")
print("Done.")
