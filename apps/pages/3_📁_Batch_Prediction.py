# pages/3_📁_Batch_Prediction.py
# Batch prediction from uploaded CSV
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_theme, brand_bar, page_header, section, footer
from alloy_info import estimate_cost

st.set_page_config(page_title="Batch Prediction — AlloyML", page_icon="▤",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
brand_bar(page_label="Batch Prediction")


@st.cache_resource
def load_models():
    try:
        clf = joblib.load(os.path.join(PROJECT_ROOT, "models/classifier_lgbm_pipeline.joblib"))
        reg = joblib.load(os.path.join(PROJECT_ROOT, "models/regressor_lgbm_multi_pipeline.joblib"))
        return clf, reg
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_resource
def load_intervals():
    try:
        return joblib.load(os.path.join(PROJECT_ROOT, "models/prediction_intervals.joblib"))
    except FileNotFoundError:
        return None


def generate_template() -> pd.DataFrame:
    required_cols = [
        "Al_wt","Cu_wt","Fe_wt","Mg_wt","Mn_wt","Si_wt",
        "Ti_wt","Zn_wt","Cr_wt","Others_wt","Density_g_cm3",
        "Series","Temper","Form","Processing",
    ]
    rows = [
        [97.9, 0.28, 0.70, 1.00, 0.15, 0.60, 0.15, 0.25, 0.20, 0.05, 2.70,
         "6xxx", "T6", "Sheet", "Wrought"],
        [90.0, 1.60, 0.50, 2.50, 0.30, 0.40, 0.20, 5.60, 0.23, 0.05, 2.81,
         "7xxx", "T6", "Sheet", "Wrought"],
        [93.5, 4.40, 0.50, 1.50, 0.60, 0.50, 0.15, 0.25, 0.10, 0.05, 2.78,
         "2xxx", "T3", "Plate", "Wrought"],
    ]
    return pd.DataFrame(rows, columns=required_cols)


page_header("Batch Prediction",
            "Upload a CSV of alloys and get predictions for every row at once.")

clf, reg = load_models()
if clf is None or reg is None:
    st.stop()

template_df = generate_template()

# ---- layout: template download + upload -----------------------------------
c1, c2 = st.columns([1, 2], gap="medium")

with c1:
    section("Step 1 · Get template")
    st.caption("Download the CSV template to ensure correct column names and order.")
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="alloy_prediction_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    section("Step 2 · Upload your data")
    st.caption("Upload a CSV with the same schema as the template.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

st.markdown("")


# ---- handle upload --------------------------------------------------------
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    input_data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

st.success(f"Loaded {len(input_data)} rows.")

with st.expander("Preview uploaded data"):
    st.dataframe(input_data, use_container_width=True)

required_cols = template_df.columns.tolist()
missing = [c for c in required_cols if c not in input_data.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.info("Please use the downloaded template format.")
    st.stop()


# ---- run predictions ------------------------------------------------------
if st.button("Run batch predictions", use_container_width=True, type="primary"):
    with st.spinner("Predicting…"):
        X = input_data[required_cols].copy()
        # derived metallurgical features (must match training)
        X["Mg_Si_ratio"] = X["Mg_wt"] / (X["Si_wt"] + 0.01)
        X["Zn_Mg_sum"]   = X["Zn_wt"] + X["Mg_wt"]
        X["Cu_Mg_sum"]   = X["Cu_wt"] + X["Mg_wt"]
        X["total_solute"] = X[["Cu_wt","Mg_wt","Mn_wt","Si_wt","Zn_wt","Cr_wt"]].sum(axis=1)

        cls_cols = [c for c in X.columns if c != "Series"]
        X_cls = X[cls_cols].copy()

        pred_classes = clf.predict(X_cls)
        try:
            pred_probs = clf.predict_proba(X_cls)
            max_probs = pred_probs.max(axis=1) * 100
        except Exception:
            max_probs = np.zeros(len(pred_classes))

        reg_targets = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
        pred_props = reg.predict(X)

        results_df = input_data.copy()
        results_df["Predicted_Family"] = pred_classes
        results_df["Confidence (%)"]   = max_probs.round(2)
        for i, target in enumerate(reg_targets):
            results_df[f"Predicted_{target}"] = pred_props[:, i].round(2)

        # Attach 90% conformal prediction intervals (absolute half-widths)
        intervals = load_intervals()
        if intervals is not None:
            for target in reg_targets:
                q90 = intervals["quantiles"][target].get(0.90)
                if q90 is not None:
                    results_df[f"PI90_{target}"] = round(q90, 2)

        # Attach cost estimate per row (based on predicted family + input Form)
        cost_rows = [estimate_cost(series, form) for series, form
                     in zip(pred_classes, results_df["Form"])]
        results_df["Cost_USD_per_kg"]   = [round(c["unit_cost_usd_kg"], 2) for c in cost_rows]
        results_df["Cost_low_USD_kg"]   = [round(c["low"], 2)  for c in cost_rows]
        results_df["Cost_high_USD_kg"]  = [round(c["high"], 2) for c in cost_rows]
        results_df["Cost_tier"]         = [c["tier"] for c in cost_rows]

    section("Results")
    preview_cols = (["Predicted_Family","Confidence (%)"] +
                    [f"Predicted_{t}" for t in reg_targets]
                    + ["Cost_USD_per_kg", "Cost_tier"])
    st.markdown(f"**Showing first 50 of {len(results_df)} rows.** "
                "Full results (including PI90 intervals and cost ranges) in the downloaded CSV.")
    st.dataframe(results_df[preview_cols].head(50), use_container_width=True)

    st.download_button(
        "Download full results (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="alloy_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

footer()
