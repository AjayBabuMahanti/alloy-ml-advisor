"""
AlloyML Research Pipeline - BTP Report Quality
===============================================
Multimodal AI for Alloy Microstructure-Property Prediction
IIT Roorkee, MIN-400A, Academic Session 2025-26

This script builds a rigorous ML pipeline for predicting:
  1. Alloy Family (Classification): from composition + processing
  2. Mechanical Properties (Regression): UTS, YS, Elongation, Hardness, Modulus

Key improvements over v1:
  - Fixed data leakage (Alloy_Series removed from classifier features)
  - Cross-validated evaluation (5-fold GroupKFold)
  - Multi-model comparison (LightGBM, RandomForest, XGBoost, SVR)
  - Hyperparameter tuning (RandomizedSearchCV)
  - Publication-quality figures saved to results/

Data Sources:
  [1] Hussey & Wilson, "Light Alloys Directory and Databook", Springer, 1998
  [2] ASM Handbook Vol. 2, "Properties and Selection: Nonferrous Alloys"
  [3] Davis, J.R., "Aluminum and Aluminum Alloys", ASM Specialty Handbook, 1993
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (GroupKFold, RandomizedSearchCV,
                                     GroupShuffleSplit, cross_val_score)
from scipy.stats import wilcoxon
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error,
                             classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'alloy_dataset_final.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Publication figure settings
matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white'
})

# Try importing optional packages
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not found, skipping XGBoost models")

try:
    from sklearn.svm import SVR
    HAS_SVR = True
except ImportError:
    HAS_SVR = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not found, skipping SHAP analysis")


# ============================================================
# SECTION A: DATA LOADING AND EDA
# ============================================================
print("=" * 60)
print("SECTION A: Data Loading and Exploratory Data Analysis")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Unique alloys: {df['Alloy'].nunique()}")
print(f"Alloy series: {sorted(df['Series'].unique())}")

# Normalize Al content so compositions sum to 100%
comp_cols = ['Al_wt', 'Cu_wt', 'Fe_wt', 'Mg_wt', 'Mn_wt',
             'Si_wt', 'Ti_wt', 'Zn_wt', 'Cr_wt', 'Others_wt']
comp_sum = df[comp_cols].sum(axis=1)
df['Al_wt'] = df['Al_wt'] - (comp_sum - 100.0)
# Clamp Al to valid range
df['Al_wt'] = df['Al_wt'].clip(lower=70.0)
comp_sum_fixed = df[comp_cols].sum(axis=1)
print(f"Composition sums after fix: {comp_sum_fixed.min():.1f} - {comp_sum_fixed.max():.1f}")

# Mark seed vs augmented
df['is_seed'] = ~df['Alloy'].str.contains('_v\\d+', regex=True)
print(f"Seed records: {df['is_seed'].sum()}, Augmented: {(~df['is_seed']).sum()}")

# ============================================================
# DERIVED METALLURGICAL FEATURES (physics-informed)
# ============================================================
# Mg2Si precipitation strength is the dominant hardening in 6xxx alloys,
# so the stoichiometric Mg/Si ratio (~1.73 for Mg2Si) is a natural feature.
df['Mg_Si_ratio'] = df['Mg_wt'] / (df['Si_wt'] + 0.01)   # +0.01 to avoid div/0
# MgZn2 (eta') and Mg3Zn3Al2 (T-phase) strengthen 7xxx alloys.
df['Zn_Mg_sum'] = df['Zn_wt'] + df['Mg_wt']
# S-phase (Al2CuMg) and theta' (Al2Cu) precipitates govern 2xxx response.
df['Cu_Mg_sum'] = df['Cu_wt'] + df['Mg_wt']
# Total solute content -- proxy for deviation from pure-Al matrix.
df['total_solute'] = df[['Cu_wt','Mg_wt','Mn_wt','Si_wt','Zn_wt','Cr_wt']].sum(axis=1)
print(f"Added derived features: Mg_Si_ratio, Zn_Mg_sum, Cu_Mg_sum, total_solute")

# ---- Dataset statistics table ----
numeric_features = ['Density_g_cm3'] + comp_cols
targets = ['UTS_MPa', 'YS_MPa', 'Elongation_pct', 'Hardness_HB', 'Modulus_GPa']

stats_list = []
for col in numeric_features + targets:
    stats_list.append({
        'Feature': col,
        'Type': 'Target' if col in targets else 'Composition' if 'wt' in col else 'Physical',
        'Count': df[col].count(),
        'Missing%': f"{df[col].isna().mean()*100:.1f}",
        'Min': f"{df[col].min():.2f}",
        'Max': f"{df[col].max():.2f}",
        'Mean': f"{df[col].mean():.2f}",
        'Std': f"{df[col].std():.2f}"
    })
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(os.path.join(RESULTS_DIR, 'dataset_statistics.csv'), index=False)
print("\nDataset statistics:")
print(stats_df.to_string(index=False))

# ---- Figure 1: Target distributions ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
target_labels = ['UTS (MPa)', 'YS (MPa)', 'Elongation (%)', 'Hardness (HB)', 'Modulus (GPa)']

for i, (col, label) in enumerate(zip(targets, target_labels)):
    ax = axes_flat[i]
    sns.histplot(df[col], kde=True, ax=ax, color='#2196F3', edgecolor='white')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.set_title(f'{label}\n(mean={df[col].mean():.1f}, std={df[col].std():.1f})')

axes_flat[5].set_visible(False)
plt.suptitle('Distribution of Mechanical Property Targets', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_target_distributions.png'))
plt.close()
print("Saved: fig_target_distributions.png")

# ---- Figure 2: Correlation heatmap ----
corr_cols = ['Density_g_cm3', 'Cu_wt', 'Mg_wt', 'Mn_wt', 'Si_wt', 'Zn_wt', 'Cr_wt'] + targets
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix: Composition Elements and Mechanical Properties',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_correlation_heatmap.png'))
plt.close()
print("Saved: fig_correlation_heatmap.png")

# ---- Figure 3: Series distribution ----
series_counts = df['Series'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2196F3' if c >= 10 else '#FF5722' for c in series_counts.values]
bars = ax.bar(series_counts.index, series_counts.values, color=colors, edgecolor='white', linewidth=0.5)
for bar, count in zip(bars, series_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(count), ha='center', fontweight='bold', fontsize=10)
ax.set_xlabel('Alloy Series')
ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Composition: Samples per Alloy Series', fontweight='bold')
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='n=10 threshold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_class_distribution.png'))
plt.close()
print("Saved: fig_class_distribution.png")


# ============================================================
# SECTION B: PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("SECTION B: Preprocessing and Feature Engineering")
print("=" * 60)

# Define feature columns (raw composition + derived metallurgical features)
numeric_cols = ['Density_g_cm3', 'Al_wt', 'Cu_wt', 'Fe_wt', 'Mg_wt',
                'Mn_wt', 'Si_wt', 'Ti_wt', 'Zn_wt', 'Cr_wt', 'Others_wt',
                'Mg_Si_ratio', 'Zn_Mg_sum', 'Cu_Mg_sum', 'total_solute']

# FOR CLASSIFICATION: Series is the target, NOT a feature
cls_cat_cols = ['Temper', 'Form', 'Processing']
cls_target = 'Series'

# FOR REGRESSION: Series IS a valid feature (we know the alloy type)
reg_cat_cols = ['Series', 'Temper', 'Form', 'Processing']

# Filter out Cast alloys for wrought-only analysis (more focused for paper)
# Keep cast for regression but use wrought-only for classification
df_wrought = df[df['Processing'] != 'Die Cast'].copy()
print(f"Wrought + cast dataset: {len(df_wrought)} samples (excluded die cast)")
print(f"\nClassification target distribution:")
print(df_wrought['Series'].value_counts().sort_index())

# Train/test split grouped by BASE alloy name (variants stay together)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
# Strip _v1, _v2 etc. so augmented variants group with their seed
groups = df_wrought['Alloy'].str.replace(r'_v\d+$', '', regex=True)
train_idx, test_idx = next(gss.split(df_wrought, groups=groups))
train_df = df_wrought.iloc[train_idx].reset_index(drop=True)
test_df = df_wrought.iloc[test_idx].reset_index(drop=True)
print(f"\nTrain: {len(train_df)} samples, Test: {len(test_df)} samples")
print(f"Train alloys: {train_df['Alloy'].nunique()}, Test alloys: {test_df['Alloy'].nunique()}")

# Prepare feature matrices
X_train_cls = train_df[numeric_cols + cls_cat_cols].copy()
X_test_cls = test_df[numeric_cols + cls_cat_cols].copy()
y_train_cls = train_df[cls_target].copy()
y_test_cls = test_df[cls_target].copy()

X_train_reg = train_df[numeric_cols + reg_cat_cols].copy()
X_test_reg = test_df[numeric_cols + reg_cat_cols].copy()
y_train_reg = train_df[targets].copy()
y_test_reg = test_df[targets].copy()

# Build preprocessors
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')

cls_preprocessor = build_preprocessor(numeric_cols, cls_cat_cols)
reg_preprocessor = build_preprocessor(numeric_cols, reg_cat_cols)


# ============================================================
# SECTION C: MODEL TRAINING WITH CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION C: Cross-Validated Model Comparison")
print("=" * 60)

# Helper: cross-validated classification evaluation
def evaluate_classifier(name, model, X, y, groups, n_splits=5):
    """Run GroupKFold CV for classification, return metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    f1_weighted = []
    f1_macro = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        f1_weighted.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
        f1_macro.append(f1_score(y_val, y_pred, average='macro', zero_division=0))

    return {
        'Model': name,
        'Accuracy': f"{np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}",
        'Weighted_F1': f"{np.mean(f1_weighted):.3f} +/- {np.std(f1_weighted):.3f}",
        'Macro_F1': f"{np.mean(f1_macro):.3f} +/- {np.std(f1_macro):.3f}",
        'acc_mean': np.mean(accuracies),
        'f1w_mean': np.mean(f1_weighted),
        'f1w_scores': f1_weighted,
        'acc_scores': accuracies
    }


# Helper: cross-validated regression evaluation per target
def evaluate_regressor(name, model_fn, preprocessor, X, y, groups, n_splits=5):
    """Run GroupKFold CV for multi-target regression."""
    gkf = GroupKFold(n_splits=n_splits)
    results = {t: {'rmse': [], 'mae': [], 'r2': []} for t in targets}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ('pre', preprocessor),
            ('reg', MultiOutputRegressor(model_fn()))
        ])
        # Clone preprocessor for each fold
        from sklearn.base import clone
        pipe_clone = clone(pipe)
        pipe_clone.fit(X_tr, y_tr)
        y_pred = pipe_clone.predict(X_val)
        y_pred_df = pd.DataFrame(y_pred, columns=targets, index=X_val.index)

        for j, t in enumerate(targets):
            mask = ~y_val[t].isna()
            if mask.sum() > 0:
                results[t]['rmse'].append(np.sqrt(mean_squared_error(y_val[t][mask], y_pred_df[t][mask])))
                results[t]['mae'].append(mean_absolute_error(y_val[t][mask], y_pred_df[t][mask]))
                results[t]['r2'].append(r2_score(y_val[t][mask], y_pred_df[t][mask]))

    summary = []
    for t in targets:
        summary.append({
            'Target': t,
            'Model': name,
            'RMSE': f"{np.mean(results[t]['rmse']):.2f} +/- {np.std(results[t]['rmse']):.2f}",
            'MAE': f"{np.mean(results[t]['mae']):.2f} +/- {np.std(results[t]['mae']):.2f}",
            'R2': f"{np.mean(results[t]['r2']):.3f} +/- {np.std(results[t]['r2']):.3f}",
            'rmse_mean': np.mean(results[t]['rmse']),
            'r2_mean': np.mean(results[t]['r2']),
            'r2_scores': results[t]['r2']
        })
    return summary


# ---- Classification Models ----
print("\n--- Classification: Alloy Series Prediction ---")
print(f"Features: {len(numeric_cols)} numeric + {len(cls_cat_cols)} categorical (NO Series leakage)")

cls_models = {
    'LightGBM': Pipeline([
        ('pre', cls_preprocessor),
        ('clf', LGBMClassifier(n_estimators=500, learning_rate=0.03,
                               max_depth=7, num_leaves=63,
                               min_child_samples=5,
                               class_weight='balanced',
                               random_state=SEED, verbose=-1))
    ]),
    'RandomForest': Pipeline([
        ('pre', cls_preprocessor),
        ('clf', RandomForestClassifier(n_estimators=500, max_depth=15,
                                       min_samples_leaf=1,
                                       class_weight='balanced',
                                       random_state=SEED))
    ]),
}

if HAS_XGB:
    # XGBoost needs numeric labels -- encode via LabelEncoder inside pipeline
    from sklearn.base import BaseEstimator, ClassifierMixin
    class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
        """Wraps XGBClassifier to handle string labels."""
        def __init__(self, **kwargs):
            self.xgb = XGBClassifier(**kwargs)
            self.le = LabelEncoder()
        def fit(self, X, y):
            self.le.fit(y)
            self.classes_ = self.le.classes_
            self.xgb.fit(X, self.le.transform(y))
            return self
        def predict(self, X):
            return self.le.inverse_transform(self.xgb.predict(X))
        def predict_proba(self, X):
            return self.xgb.predict_proba(X)

    cls_models['XGBoost'] = Pipeline([
        ('pre', cls_preprocessor),
        ('clf', XGBClassifierWrapper(n_estimators=300, learning_rate=0.05,
                                     max_depth=5, eval_metric='mlogloss',
                                     random_state=SEED))
    ])

cls_results = []
cls_groups = train_df['Alloy'].str.replace(r'_v\d+$', '', regex=True)
for name, pipe in cls_models.items():
    t0 = time.time()
    result = evaluate_classifier(name, pipe, X_train_cls, y_train_cls, cls_groups)
    elapsed = time.time() - t0
    cls_results.append(result)
    print(f"  {name:15s} | Acc={result['Accuracy']} | F1w={result['Weighted_F1']} | {elapsed:.1f}s")

cls_comparison = pd.DataFrame([{k: v for k, v in r.items()
                                 if k in ['Model', 'Accuracy', 'Weighted_F1', 'Macro_F1']}
                                for r in cls_results])
cls_comparison.to_csv(os.path.join(RESULTS_DIR, 'table_classification_comparison.csv'), index=False)
print("\nSaved: table_classification_comparison.csv")

# ---- Statistical significance between classifiers (Wilcoxon signed-rank) ----
print("\n--- Wilcoxon Signed-Rank Tests (per-fold F1_weighted) ---")
cls_sig_rows = []
for i in range(len(cls_results)):
    for j in range(i + 1, len(cls_results)):
        a, b = cls_results[i], cls_results[j]
        try:
            stat, p = wilcoxon(a['f1w_scores'], b['f1w_scores'])
            verdict = 'significant' if p < 0.05 else 'n.s.'
        except ValueError:
            stat, p, verdict = np.nan, np.nan, 'tied'
        print(f"  {a['Model']:12s} vs {b['Model']:12s} | p={p:.3f} | {verdict}")
        cls_sig_rows.append({'Model_A': a['Model'], 'Model_B': b['Model'],
                             'p_value': f"{p:.3f}", 'Significant_a=0.05': verdict})
pd.DataFrame(cls_sig_rows).to_csv(
    os.path.join(RESULTS_DIR, 'table_classification_wilcoxon.csv'), index=False)
print("Saved: table_classification_wilcoxon.csv")

# Select best classifier
best_cls_idx = np.argmax([r['f1w_mean'] for r in cls_results])
best_cls_name = cls_results[best_cls_idx]['Model']
best_cls_pipe = cls_models[best_cls_name]
print(f"\nBest classifier: {best_cls_name}")

# ---- Hyperparameter tuning on best classifier (RandomizedSearchCV) ----
CLS_PARAM_GRIDS = {
    'LightGBM': {
        'clf__n_estimators': [300, 500, 800],
        'clf__learning_rate': [0.02, 0.03, 0.05, 0.08],
        'clf__num_leaves': [31, 63, 127],
        'clf__max_depth': [5, 7, -1],
        'clf__min_child_samples': [3, 5, 10],
    },
    'RandomForest': {
        'clf__n_estimators': [300, 500, 800],
        'clf__max_depth': [10, 15, 20, None],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', 0.5],
    },
    'XGBoost': {
        'clf__n_estimators': [300, 500, 800],
        'clf__learning_rate': [0.02, 0.03, 0.05, 0.08],
        'clf__max_depth': [3, 5, 7],
        'clf__min_child_weight': [1, 3, 5],
    },
}
print(f"\n--- Hyperparameter Tuning: {best_cls_name} (RandomizedSearchCV, n_iter=15) ---")
t0 = time.time()
param_grid_cls = CLS_PARAM_GRIDS.get(best_cls_name, {})
if param_grid_cls:
    search_cls = RandomizedSearchCV(
        best_cls_pipe, param_grid_cls, n_iter=15,
        cv=GroupKFold(n_splits=3), scoring='f1_weighted',
        random_state=SEED, n_jobs=-1, verbose=0
    )
    search_cls.fit(X_train_cls, y_train_cls, groups=cls_groups)
    best_clf = search_cls.best_estimator_
    print(f"  Best params: {search_cls.best_params_}")
    print(f"  Tuned F1w (inner CV): {search_cls.best_score_:.3f}")
else:
    from sklearn.base import clone as sk_clone
    best_clf = sk_clone(best_cls_pipe)
    best_clf.fit(X_train_cls, y_train_cls)
print(f"  Tuning elapsed: {time.time()-t0:.1f}s")

# Evaluate tuned classifier on test
y_pred_cls = best_clf.predict(X_test_cls)
print(f"\nTest set classification report (tuned {best_cls_name}):")
print(classification_report(y_test_cls, y_pred_cls, zero_division=0))


# ---- Regression Models ----
print("\n--- Regression: Mechanical Property Prediction ---")
print(f"Features: {len(numeric_cols)} numeric + {len(reg_cat_cols)} categorical (Series included)")

reg_model_fns = {
    'LightGBM': lambda: LGBMRegressor(n_estimators=800, learning_rate=0.02,
                                       max_depth=7, num_leaves=63,
                                       min_child_samples=3,
                                       random_state=SEED, verbose=-1),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=500, max_depth=20,
                                                   min_samples_leaf=1,
                                                   random_state=SEED),
}

if HAS_XGB:
    reg_model_fns['XGBoost'] = lambda: XGBRegressor(n_estimators=800, learning_rate=0.02,
                                                     max_depth=7, random_state=SEED)

all_reg_results = []
reg_groups = train_df['Alloy'].str.replace(r'_v\d+$', '', regex=True)
for name, model_fn in reg_model_fns.items():
    t0 = time.time()
    results = evaluate_regressor(name, model_fn, build_preprocessor(numeric_cols, reg_cat_cols),
                                 X_train_reg, y_train_reg, reg_groups)
    all_reg_results.extend(results)
    elapsed = time.time() - t0
    avg_r2 = np.mean([r['r2_mean'] for r in results])
    print(f"  {name:15s} | Avg R2={avg_r2:.3f} | {elapsed:.1f}s")

reg_comparison = pd.DataFrame([{k: v for k, v in r.items()
                                 if k in ['Target', 'Model', 'RMSE', 'MAE', 'R2']}
                                for r in all_reg_results])
reg_comparison.to_csv(os.path.join(RESULTS_DIR, 'table_regression_comparison.csv'), index=False)
print("\nSaved: table_regression_comparison.csv")

# ---- Statistical significance between regressors (per-target Wilcoxon on R2) ----
print("\n--- Wilcoxon Signed-Rank Tests (regressor pairs, per target, R2 per fold) ---")
reg_sig_rows = []
models_in_reg = list(reg_model_fns.keys())
by_target_model = {}
for r in all_reg_results:
    by_target_model[(r['Target'], r['Model'])] = r['r2_scores']
for t in targets:
    for i in range(len(models_in_reg)):
        for j in range(i + 1, len(models_in_reg)):
            a_name, b_name = models_in_reg[i], models_in_reg[j]
            a_scores = by_target_model.get((t, a_name), [])
            b_scores = by_target_model.get((t, b_name), [])
            try:
                stat, p = wilcoxon(a_scores, b_scores)
                verdict = 'significant' if p < 0.05 else 'n.s.'
            except ValueError:
                stat, p, verdict = np.nan, np.nan, 'tied'
            reg_sig_rows.append({'Target': t, 'Model_A': a_name, 'Model_B': b_name,
                                 'p_value': f"{p:.3f}", 'Significant_a=0.05': verdict})
            print(f"  {t:18s} | {a_name:12s} vs {b_name:12s} | p={p:.3f} | {verdict}")
pd.DataFrame(reg_sig_rows).to_csv(
    os.path.join(RESULTS_DIR, 'table_regression_wilcoxon.csv'), index=False)
print("Saved: table_regression_wilcoxon.csv")

# Select best regressor (by average R2 across targets)
model_r2_avg = {}
for r in all_reg_results:
    if r['Model'] not in model_r2_avg:
        model_r2_avg[r['Model']] = []
    model_r2_avg[r['Model']].append(r['r2_mean'])
model_r2_avg = {k: np.mean(v) for k, v in model_r2_avg.items()}
best_reg_name = max(model_r2_avg, key=model_r2_avg.get)
print(f"\nBest regressor: {best_reg_name} (avg R2={model_r2_avg[best_reg_name]:.3f})")

# ---- Hyperparameter tuning on best regressor (RandomizedSearchCV) ----
REG_PARAM_GRIDS = {
    'LightGBM': {
        'reg__estimator__n_estimators': [500, 800, 1200],
        'reg__estimator__learning_rate': [0.01, 0.02, 0.03, 0.05],
        'reg__estimator__num_leaves': [31, 63, 127],
        'reg__estimator__max_depth': [5, 7, -1],
        'reg__estimator__min_child_samples': [3, 5, 10],
    },
    'RandomForest': {
        'reg__estimator__n_estimators': [300, 500, 800],
        'reg__estimator__max_depth': [15, 20, 25, None],
        'reg__estimator__min_samples_leaf': [1, 2, 3],
        'reg__estimator__max_features': ['sqrt', 'log2', 0.5],
    },
    'XGBoost': {
        'reg__estimator__n_estimators': [500, 800, 1200],
        'reg__estimator__learning_rate': [0.01, 0.02, 0.05],
        'reg__estimator__max_depth': [5, 7, 9],
        'reg__estimator__min_child_weight': [1, 3, 5],
    },
}
print(f"\n--- Hyperparameter Tuning: {best_reg_name} Regressor (RandomizedSearchCV, n_iter=10) ---")
t0 = time.time()
param_grid_reg = REG_PARAM_GRIDS.get(best_reg_name, {})
tuning_reg_pipe = Pipeline([
    ('pre', reg_preprocessor),
    ('reg', MultiOutputRegressor(reg_model_fns[best_reg_name]()))
])
if param_grid_reg:
    search_reg = RandomizedSearchCV(
        tuning_reg_pipe, param_grid_reg, n_iter=10,
        cv=GroupKFold(n_splits=3), scoring='r2',
        random_state=SEED, n_jobs=-1, verbose=0, refit=True
    )
    search_reg.fit(X_train_reg, y_train_reg, groups=reg_groups)
    best_reg = search_reg.best_estimator_
    print(f"  Best params: {search_reg.best_params_}")
    print(f"  Tuned R2 (inner CV, avg across targets): {search_reg.best_score_:.3f}")
else:
    best_reg = tuning_reg_pipe
    best_reg.fit(X_train_reg, y_train_reg)
print(f"  Tuning elapsed: {time.time()-t0:.1f}s")

y_pred_reg = best_reg.predict(X_test_reg)
y_pred_reg_df = pd.DataFrame(y_pred_reg, columns=targets, index=X_test_reg.index)

print(f"\nTest set regression metrics ({best_reg_name}):")
for t in targets:
    mask = ~y_test_reg[t].isna()
    rmse = np.sqrt(mean_squared_error(y_test_reg[t][mask], y_pred_reg_df[t][mask]))
    mae = mean_absolute_error(y_test_reg[t][mask], y_pred_reg_df[t][mask])
    r2 = r2_score(y_test_reg[t][mask], y_pred_reg_df[t][mask])
    print(f"  {t:20s} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f}")


# ============================================================
# SECTION D: PUBLICATION-QUALITY FIGURES
# ============================================================
print("\n" + "=" * 60)
print("SECTION D: Generating Publication-Quality Figures")
print("=" * 60)

# ---- Figure 4: Confusion Matrix ----
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=sorted(y_test_cls.unique()))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test_cls.unique()),
            yticklabels=sorted(y_test_cls.unique()), ax=ax,
            square=True, linewidths=0.5)
ax.set_xlabel('Predicted Alloy Series', fontsize=12)
ax.set_ylabel('True Alloy Series', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_cls_name} Classifier\n(Test Set, n={len(y_test_cls)})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_confusion_matrix.png'))
plt.close()
print("Saved: fig_confusion_matrix.png")

# ---- Figure 5: Actual vs Predicted (THE KEY FIGURE) ----
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes_flat = axes.flatten()

for i, (col, label) in enumerate(zip(targets, target_labels)):
    ax = axes_flat[i]
    mask = ~y_test_reg[col].isna()
    actual = y_test_reg[col][mask].values
    predicted = y_pred_reg_df[col][mask].values

    ax.scatter(actual, predicted, alpha=0.6, s=50, c='#2196F3', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    lims = [min(actual.min(), predicted.min()) * 0.95,
            max(actual.max(), predicted.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction', alpha=0.8)

    # R2 annotation
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ax.annotate(f'R\u00b2 = {r2:.3f}\nRMSE = {rmse:.1f}',
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=11,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='wheat', alpha=0.8))
    ax.set_xlabel(f'Actual {label}', fontsize=11)
    ax.set_ylabel(f'Predicted {label}', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

axes_flat[5].set_visible(False)
plt.suptitle(f'Actual vs Predicted Mechanical Properties ({best_reg_name})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_actual_vs_predicted.png'))
plt.close()
print("Saved: fig_actual_vs_predicted.png")

# ---- Figure 6: Residual Plots ----
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes_flat = axes.flatten()

for i, (col, label) in enumerate(zip(targets, target_labels)):
    ax = axes_flat[i]
    mask = ~y_test_reg[col].isna()
    actual = y_test_reg[col][mask].values
    predicted = y_pred_reg_df[col][mask].values
    residuals = actual - predicted

    ax.scatter(predicted, residuals, alpha=0.6, s=50, c='#FF5722', edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Predicted {label}', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')

axes_flat[5].set_visible(False)
plt.suptitle('Residual Analysis: Prediction Errors', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_residual_plots.png'))
plt.close()
print("Saved: fig_residual_plots.png")

# ---- Figure 7: Feature Importance ----
try:
    # Get feature names from fitted preprocessor
    reg_pre_fitted = best_reg.named_steps['pre']
    ohe = reg_pre_fitted.named_transformers_['cat'].named_steps['ohe']
    feature_names = numeric_cols + list(ohe.get_feature_names_out(reg_cat_cols))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for i, (col, label) in enumerate(zip(targets, target_labels)):
        ax = axes_flat[i]
        estimator = best_reg.named_steps['reg'].estimators_[i]
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            top_n = min(15, len(importances))
            idx = np.argsort(importances)[-top_n:]
            ax.barh(range(len(idx)), importances[idx], color='#4CAF50', edgecolor='white')
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([feature_names[j] if j < len(feature_names) else f'feat_{j}'
                                for j in idx], fontsize=9)
            ax.set_title(f'{label}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Importance')

    axes_flat[5].set_visible(False)
    plt.suptitle(f'Feature Importance ({best_reg_name}) - Top 15 per Target',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_feature_importance.png'))
    plt.close()
    print("Saved: fig_feature_importance.png")
except Exception as e:
    print(f"WARNING: Could not generate feature importance plot: {e}")

# ---- Figure 8: Model Comparison Bar Chart ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification comparison
cls_names = [r['Model'] for r in cls_results]
cls_accs = [r['acc_mean'] for r in cls_results]
cls_f1s = [r['f1w_mean'] for r in cls_results]

ax = axes[0]
x = np.arange(len(cls_names))
w = 0.35
ax.bar(x - w/2, cls_accs, w, label='Accuracy', color='#2196F3', edgecolor='white')
ax.bar(x + w/2, cls_f1s, w, label='Weighted F1', color='#FF9800', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(cls_names)
ax.set_ylabel('Score')
ax.set_title('Classification Model Comparison', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend()
for xi, (a, f) in enumerate(zip(cls_accs, cls_f1s)):
    ax.text(xi - w/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=9)
    ax.text(xi + w/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=9)

# Regression comparison (avg R2)
ax = axes[1]
reg_names = list(model_r2_avg.keys())
reg_r2s = [model_r2_avg[n] for n in reg_names]
bars = ax.bar(reg_names, reg_r2s, color='#4CAF50', edgecolor='white')
ax.set_ylabel('Average R\u00b2')
ax.set_title('Regression Model Comparison (Avg R\u00b2)', fontweight='bold')
ax.set_ylim(0, 1.1)
for bar, r2 in zip(bars, reg_r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{r2:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_model_comparison.png'))
plt.close()
print("Saved: fig_model_comparison.png")

# ---- SHAP Analysis ----
if HAS_SHAP:
    try:
        print("\nGenerating SHAP analysis (UTS)...")
        X_test_transformed = reg_pre_fitted.transform(X_test_reg)
        uts_estimator = best_reg.named_steps['reg'].estimators_[0]  # UTS
        explainer = shap.TreeExplainer(uts_estimator)
        shap_values = explainer.shap_values(X_test_transformed)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_transformed,
                          feature_names=feature_names, show=False, max_display=15)
        plt.title('SHAP Feature Importance - UTS Prediction', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'fig_shap_summary_uts.png'))
        plt.close()
        print("Saved: fig_shap_summary_uts.png")
    except Exception as e:
        print(f"WARNING: SHAP analysis failed: {e}")


# ============================================================
# SECTION E: SAVE MODELS FOR WEB APP
# ============================================================
print("\n" + "=" * 60)
print("SECTION E: Saving Production Models")
print("=" * 60)

# Save classifier
joblib.dump(best_clf, os.path.join(MODELS_DIR, 'classifier_lgbm_pipeline.joblib'))
print(f"Saved classifier: {best_cls_name}")

# Save regressor
joblib.dump(best_reg, os.path.join(MODELS_DIR, 'regressor_lgbm_multi_pipeline.joblib'))
print(f"Saved regressor: {best_reg_name}")

# Save preprocessor (for web app)
joblib.dump(reg_preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
print("Saved preprocessor")

# Save metadata for the web app
metadata = {
    'cls_features_numeric': numeric_cols,
    'cls_features_cat': cls_cat_cols,
    'reg_features_numeric': numeric_cols,
    'reg_features_cat': reg_cat_cols,
    'targets': targets,
    'target_labels': target_labels,
    'best_cls_model': best_cls_name,
    'best_reg_model': best_reg_name,
    'dataset_size': len(df_wrought),
    'train_size': len(train_df),
    'test_size': len(test_df),
}
joblib.dump(metadata, os.path.join(MODELS_DIR, 'metadata.joblib'))
print("Saved metadata")


# ============================================================
# SECTION F: ABLATION STUDY -- seed-only vs seed+augmented
# ============================================================
# Question: does augmentation actually help, or does the model just memorize noise?
# Protocol: re-run the best classifier and regressor on seed records ONLY (no _vN),
#           using the same GroupKFold CV, and compare against seed+augmented results.
print("\n" + "=" * 60)
print("SECTION F: Ablation Study (seed-only vs seed+augmented)")
print("=" * 60)

# Filter to seed records only (exclude _v1, _v2, ... augmented variants)
df_seed_only = df_wrought[df_wrought['is_seed']].copy().reset_index(drop=True)
print(f"Seed-only subset: {len(df_seed_only)} records "
      f"(vs {len(df_wrought)} in seed+augmented)")
print(f"Seed-only series distribution:\n{df_seed_only['Series'].value_counts().sort_index()}")

# Build the same features on the seed-only subset
X_seed_cls = df_seed_only[numeric_cols + cls_cat_cols].copy()
y_seed_cls = df_seed_only[cls_target].copy()
X_seed_reg = df_seed_only[numeric_cols + reg_cat_cols].copy()
y_seed_reg = df_seed_only[targets].copy()
seed_groups = df_seed_only['Alloy'].str.replace(r'_v\d+$', '', regex=True)

# Re-evaluate the SAME models on seed-only (using untuned pipelines for fair comparison)
print("\n--- Seed-only classification CV ---")
seed_cls_results = []
for name, pipe in cls_models.items():
    # Recreate clean copy (cls_models were already fit during initial CV)
    from sklearn.base import clone as sk_clone_f
    fresh_pipe = sk_clone_f(pipe)
    try:
        result = evaluate_classifier(name, fresh_pipe, X_seed_cls, y_seed_cls, seed_groups, n_splits=5)
        seed_cls_results.append(result)
        print(f"  {name:15s} | Acc={result['Accuracy']} | F1w={result['Weighted_F1']}")
    except Exception as e:
        print(f"  {name:15s} | SKIPPED ({e})")

print("\n--- Seed-only regression CV ---")
seed_reg_results = []
for name, model_fn in reg_model_fns.items():
    try:
        results = evaluate_regressor(name, model_fn,
                                     build_preprocessor(numeric_cols, reg_cat_cols),
                                     X_seed_reg, y_seed_reg, seed_groups, n_splits=5)
        seed_reg_results.extend(results)
        avg_r2 = np.mean([r['r2_mean'] for r in results])
        print(f"  {name:15s} | Avg R2={avg_r2:.3f}")
    except Exception as e:
        print(f"  {name:15s} | SKIPPED ({e})")

# Build ablation comparison table
print("\n--- Ablation Summary (seed-only vs seed+augmented, Weighted F1 / avg R2) ---")
ablation_rows = []
# Classification rows
for full, seed in zip(cls_results, seed_cls_results):
    if full['Model'] == seed['Model']:
        ablation_rows.append({
            'Task': 'Classification (F1w)',
            'Model': full['Model'],
            'Seed+Aug_mean': f"{full['f1w_mean']:.3f}",
            'Seed-only_mean': f"{seed['f1w_mean']:.3f}",
            'Delta': f"{full['f1w_mean']-seed['f1w_mean']:+.3f}",
            'N_seed_aug': len(train_df),
            'N_seed_only': len(df_seed_only),
        })
# Regression rows (averaged across targets per model)
full_by_model = {}
seed_by_model = {}
for r in all_reg_results:
    full_by_model.setdefault(r['Model'], []).append(r['r2_mean'])
for r in seed_reg_results:
    seed_by_model.setdefault(r['Model'], []).append(r['r2_mean'])
for m in full_by_model:
    if m in seed_by_model:
        fm = np.mean(full_by_model[m])
        sm = np.mean(seed_by_model[m])
        ablation_rows.append({
            'Task': 'Regression (avg R2)',
            'Model': m,
            'Seed+Aug_mean': f"{fm:.3f}",
            'Seed-only_mean': f"{sm:.3f}",
            'Delta': f"{fm-sm:+.3f}",
            'N_seed_aug': len(train_df),
            'N_seed_only': len(df_seed_only),
        })
ablation_df = pd.DataFrame(ablation_rows)
print(ablation_df.to_string(index=False))
ablation_df.to_csv(os.path.join(RESULTS_DIR, 'table_ablation_seed_vs_augmented.csv'), index=False)
print("Saved: table_ablation_seed_vs_augmented.csv")

# Per-target regression ablation (finer detail for the paper)
per_target_rows = []
full_pt = {(r['Target'], r['Model']): r['r2_mean'] for r in all_reg_results}
seed_pt = {(r['Target'], r['Model']): r['r2_mean'] for r in seed_reg_results}
for (tgt, mdl), fr in full_pt.items():
    sr = seed_pt.get((tgt, mdl))
    if sr is not None:
        per_target_rows.append({
            'Target': tgt, 'Model': mdl,
            'R2_seed_aug': f"{fr:.3f}",
            'R2_seed_only': f"{sr:.3f}",
            'Delta': f"{fr-sr:+.3f}",
        })
pd.DataFrame(per_target_rows).to_csv(
    os.path.join(RESULTS_DIR, 'table_ablation_per_target.csv'), index=False)
print("Saved: table_ablation_per_target.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nDataset: {len(df)} alloys ({df['Series'].nunique()} series)")
print(f"Data sources: Light Alloys Databook [1], ASM Handbook [2,3]")
print(f"\nBest Classifier: {best_cls_name}")
print(f"  Test Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.3f}")
print(f"  Test Weighted F1: {f1_score(y_test_cls, y_pred_cls, average='weighted', zero_division=0):.3f}")
print(f"\nBest Regressor: {best_reg_name}")
for t in targets:
    mask = ~y_test_reg[t].isna()
    r2 = r2_score(y_test_reg[t][mask], y_pred_reg_df[t][mask])
    print(f"  {t:20s} R2={r2:.3f}")

print(f"\nFigures saved to: {RESULTS_DIR}/")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  {f}")

print("\nAll done! Models ready for Streamlit app.")
