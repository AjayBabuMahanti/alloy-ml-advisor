# AlloyML — AI-Powered Aluminum Alloy Advisor

**Final Year B.Tech Project (MIN-400A, 2025–26)** — IIT Roorkee, Mechanical Engineering
Under Prof. Anuj Bisht

**Team:** Akshat Jain · Alumolu Rakesh Reddy · Mahanti Ajay Babu · Himanshu Panwar · Ashok Godara

---

## What This App Does

A Streamlit web application that helps engineers choose the right aluminum alloy for their application. Four integrated tools:

| Page | Purpose |
|---|---|
| **⚙️ Forward Prediction** | Enter composition + processing → predict mechanical properties + alloy family |
| **🔍 Inverse Prediction** | Enter target properties → (a) find closest real alloys from the database, or (b) generate a candidate composition via scipy optimization |
| **📁 Batch Prediction** | Upload a CSV of alloys → bulk predictions for all of them |
| **🤖 AI Assistant** | Chat with a materials-engineering LLM (Mistral) for alloy selection guidance, tradeoffs, and application advice |

The models are trained on a curated dataset of **783 real aluminum alloy records** compiled from ASM Handbook, Light Alloys Databook, and Davis's ASM Specialty Handbook, extended with physics-informed augmentation.

---

## Project Structure

```
AlloyMLProject/
├── README.md                 — This file
├── requirements.txt          — Python dependencies (Python 3.12)
├── launch_app.ps1            — PowerShell launcher for the web app
├── .gitignore                — Git exclusions (venv, secrets, cache)
├── .streamlit/
│   └── secrets.toml          — Mistral API key (git-ignored; never commit)
│
├── data/
│   ├── build_dataset.py            — Compiles 129 seed records from Refs [1–3]
│   ├── alloy_dataset_v2.csv        — 129 seed records (real, handbook-sourced)
│   ├── augment_dataset.py          — Physics-informed augmentation (×2–5 per seed)
│   ├── alloy_dataset_augmented.csv — 425 seed + augmented records
│   ├── build_full_dataset.py       — Expanded seeds + augmentation → 783 records
│   └── alloy_dataset_final.csv     — THE training dataset (783 records)
│
├── src/
│   └── alloy_ml_research.py  — Training pipeline (CV + HP tuning + ablation + figures)
│
├── apps/                     — Streamlit multi-page web app
│   ├── app.py                — Home page
│   ├── styles.py             — Shared CSS / plotly theme
│   └── pages/
│       ├── 1_⚙️_Forward_Prediction.py   — Composition → properties + PDF datasheet
│       ├── 2_🔍_Inverse_Prediction.py   — Target properties → (find OR generate)
│       ├── 3_📁_Batch_Prediction.py     — CSV upload for bulk predictions
│       └── 4_🤖_AI_Assistant.py         — Mistral LLM chatbot for alloy guidance
│
├── models/                   — Trained joblib pipelines (produced by src/alloy_ml_research.py)
│   ├── classifier_lgbm_pipeline.joblib
│   ├── regressor_lgbm_multi_pipeline.joblib
│   ├── preprocessor.joblib
│   └── metadata.joblib
│
├── results/                  — Publication figures + comparison tables
│   ├── fig_target_distributions.png
│   ├── fig_correlation_heatmap.png
│   ├── fig_class_distribution.png
│   ├── fig_confusion_matrix.png
│   ├── fig_actual_vs_predicted.png
│   ├── fig_residual_plots.png
│   ├── fig_feature_importance.png
│   ├── fig_model_comparison.png
│   ├── fig_shap_summary_uts.png
│   ├── dataset_statistics.csv
│   ├── table_classification_comparison.csv
│   ├── table_classification_wilcoxon.csv      — NEW: stat sig tests
│   ├── table_regression_comparison.csv
│   ├── table_regression_wilcoxon.csv          — NEW: stat sig tests
│   ├── table_ablation_seed_vs_augmented.csv   — NEW: ablation summary
│   └── table_ablation_per_target.csv          — NEW: per-target ablation
│
└── venv/                     — Python 3.12 virtual environment
```

---

## Dataset

**Final training dataset:** 783 records, 10 composition features + density + 4 categorical variables (Series, Temper, Form, Processing) + 5 mechanical-property targets.

**Sources for seed data:**
1. Hussey, R.J. & Wilson, J. — *Light Alloys Directory and Databook*, Springer, 1998.
2. ASM Handbook Vol. 2 — *Properties and Selection: Nonferrous Alloys and Special-Purpose Materials.*
3. Davis, J.R. — *Aluminum and Aluminum Alloys*, ASM Specialty Handbook, 1993.
4. Hatch, J.E. — *Aluminum: Properties and Physical Metallurgy*, ASM, 1984.
5. Kaufman, J.G. — *Properties of Aluminum Alloys*, ASM International, 1999.

**Augmentation methodology** (see [data/augment_dataset.py](data/augment_dataset.py)):
- Composition perturbations bounded by AA specification tolerance bands.
- Property adjustments via linearized solid-solution strengthening coefficients from Hatch [4].
- Gaussian noise per ASTM E8 / E10 reproducibility (σ = 2–5%).
- Physical constraints enforced: YS ≤ 0.98·UTS, composition sums = 100%, modulus clipped to [65, 82] GPa.

---

## Features & Engineering

**15 numeric features** per record:
- 11 raw: `Density_g_cm3` + 10 composition wt% (Al, Cu, Fe, Mg, Mn, Si, Ti, Zn, Cr, Others)
- 4 derived metallurgical descriptors (physics-informed):
  - `Mg_Si_ratio` — targets Mg₂Si precipitation in 6xxx alloys (stoichiometric at ~1.73)
  - `Zn_Mg_sum` — proxy for η' (MgZn₂) strengthening in 7xxx
  - `Cu_Mg_sum` — proxy for S-phase (Al₂CuMg) in 2xxx
  - `total_solute` — total alloying content, proxy for matrix deviation from pure Al

**4 categorical features** (one-hot encoded): Series, Temper, Form, Processing.
(Series excluded from classifier features to prevent label leakage.)

## Results

### Classification — Alloy Series (9 classes)
*5-fold GroupKFold CV (alloy-grouped to prevent leakage); `Series` excluded from features.*

| Model | CV Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| **RandomForest** ⭐ | **0.842 ± 0.137** | **0.834 ± 0.152** | **0.788 ± 0.096** |
| LightGBM | 0.778 ± 0.119 | 0.783 ± 0.127 | 0.686 ± 0.019 |
| XGBoost | 0.717 ± 0.107 | 0.706 ± 0.139 | 0.635 ± 0.090 |

**Wilcoxon signed-rank tests (per-fold F1w):** no pairwise difference reaches p<0.05 on 5 folds — differences are directionally consistent but statistically underpowered at this fold count.

After RandomizedSearchCV (n_iter=15, inner 3-fold GroupKFold):
- Best RF params: `n_estimators=300, max_depth=15, min_samples_leaf=2, max_features='sqrt'`
- Inner-CV F1w of tuned model: 0.916 (optimistic — inner CV is 3-fold, lower variance)
- **Test-set performance:** Accuracy 0.574, Weighted F1 0.464 — much lower than CV.
  The gap reflects the group-split protocol: the test set contains 52 *alloy families* never seen in training, including entire tempers where minority classes (8xxx, 3xxx) are missed. **This is the honest generalization number and should be reported as such in the paper.**

### Regression — Mechanical Properties (5-fold GroupKFold CV)

| Target | LightGBM R² | RandomForest R² | XGBoost R² |
|---|---|---|---|
| UTS (MPa) | 0.790 ± 0.18 | 0.813 ± 0.11 | **0.820 ± 0.11** |
| Yield Strength (MPa) | 0.743 ± 0.14 | **0.757 ± 0.12** | 0.707 ± 0.15 |
| Elongation (%) | 0.793 ± 0.07 | **0.805 ± 0.07** | 0.780 ± 0.08 |
| Hardness (HB) | 0.794 ± 0.10 | **0.820 ± 0.07** | 0.790 ± 0.09 |
| Young's Modulus (GPa) | **0.363 ± 0.51** | 0.258 ± 0.65 | 0.269 ± 0.60 |
| **Avg R²** | **0.697** | 0.691 | 0.673 |

**Wilcoxon signed-rank tests:** all regressor pairs non-significant on 5 folds (smallest p = 0.188 for RF vs XGB on UTS). Models are statistically indistinguishable at this sample size — **report effect sizes alongside tests.**

**Tuned LightGBM regressor** (RandomizedSearchCV n_iter=10, inner 3-fold):
- Best params: `n_estimators=500, learning_rate=0.01, num_leaves=31, max_depth=5, min_child_samples=5`
- Test-set R²: **UTS 0.845 · YS 0.875 · Elongation 0.862 · Hardness 0.887 · Modulus 0.755**

### Ablation — Seed-Only (n=199) vs Seed+Augmented (n=775)

| Task | Model | Seed+Aug | Seed-only | Δ |
|---|---|---|---|---|
| Classification F1w | RandomForest | 0.834 | **0.861** | −0.027 |
| Classification F1w | LightGBM | 0.783 | **0.792** | −0.009 |
| Classification F1w | XGBoost | 0.706 | **0.772** | −0.067 |
| Regression avg R² | LightGBM | **0.697** | 0.634 | +0.063 |
| Regression avg R² | RandomForest | 0.691 | **0.709** | −0.018 |
| Regression avg R² | XGBoost | **0.673** | 0.666 | +0.008 |

**Interpretation:** augmentation **hurts classification slightly** (adds noise on a task that mostly wants clean category boundaries) but **helps LightGBM regression** noticeably (+0.06). The effect is target-dependent: augmentation helps Modulus prediction for LGBM by +0.33 R², nothing for RF. This mixed picture should be reported honestly in the paper as evidence that physics-informed augmentation is beneficial where signal is weak but not a free lunch.

---

## Setup

### 1. Create a virtual environment and install dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure the Mistral API key
Create `.streamlit/secrets.toml` (already git-ignored) with:
```toml
[mistral]
api_key = "YOUR_MISTRAL_KEY_HERE"
model   = "ministral-8b-latest"
```
Get a free key at [console.mistral.ai](https://console.mistral.ai).

### 3. (Optional) Retrain the models
The trained models are already in `models/`. If you want to retrain:
```powershell
python src/alloy_ml_research.py
```
This runs CV, HP tuning, ablation, saves figures, and writes fresh `models/*.joblib`.

### 4. Launch the app
```powershell
.\launch_app.ps1
# or manually:
streamlit run apps/app.py
```
App opens at http://localhost:8501.

---

## How the Inverse Prediction Works

The Inverse page offers two modes:

1. **Find from database (default):** normalized-distance nearest-neighbor over the 783-record dataset, returning the top 10 real alloys closest to the target properties.
2. **Generate new composition:** uses `scipy.optimize.differential_evolution` with the forward model as objective — searches the composition space (11 variables: 10 wt% elements + density) subject to `sum(wt%) ≈ 100` and per-element bounds, returning a synthesized candidate. Takes ~30–60s per run.

Use (1) when you want a real, specifiable alloy you can actually buy. Use (2) when you want a theoretical composition that would match your exact target, and you'll cross-reference it against (1) as a sanity check.

---

## The AI Assistant

The AI Assistant page uses the Mistral API (`ministral-8b-latest` by default). Its system prompt is grounded with:
- A summary of every alloy series in the dataset (counts, UTS ranges, typical applications).
- The user's question + conversation history.
- Guardrails: must recommend specific AA designations (e.g. 6061-T6), must be honest about tradeoffs, must not invent non-standard alloys.

Example prompts that work well:
- *"I need a lightweight alloy for a bicycle frame — what do you recommend?"*
- *"Compare 6061-T6 and 7075-T6 for an aerospace bracket."*
- *"Which aluminum alloy is best for marine applications?"*
- *"I need ~350 MPa UTS with good weldability. Options?"*

---

## Model Quality (reference)

Trained on 580 records, tested on 195 records (GroupShuffleSplit by alloy family). Tuned LightGBM test R²:

| Target | R² | Example (6061-T6) |
|---|---|---|
| UTS | 0.845 | predicted 307 vs actual 310 MPa |
| Yield Strength | 0.875 | predicted 262 vs actual 276 MPa |
| Elongation | 0.862 | predicted 11.8% vs actual 12% |
| Hardness | 0.887 | predicted 94 vs actual 95 HB |
| Modulus | 0.755 | predicted 69 vs actual 69 GPa |

Classification of alloy series uses RandomForest (84% CV accuracy). Full CV and ablation results in [results/](results/).

---

## Roadmap

- [x] Real-data dataset from handbooks (no synthetic fake-code data)
- [x] Leakage-free classification + regression pipeline
- [x] Derived metallurgical features + HP tuning
- [x] Forward / Inverse / Batch / AI Assistant web UI
- [x] Mistral LLM integration
- [ ] OOD warning when user inputs composition far from training data
- [ ] Prediction intervals (quantile regression)
- [ ] Conversation memory across chatbot + prediction pages (cross-page context)

---

## Notes

- **Inverse optimization caveat:** The generated composition is a ML-model recommendation. For production use, always cross-check against handbook values and the nearest real alloys from the database.
- **Modulus has the widest error band.** Young's modulus spans only 66–82 GPa across all aluminum alloys and is dominated by the Al matrix. The model sometimes predicts values tight against this range.
- **API key security:** `.streamlit/secrets.toml` is git-ignored. Never commit it. If a key leaks, rotate it at [console.mistral.ai](https://console.mistral.ai) immediately.
