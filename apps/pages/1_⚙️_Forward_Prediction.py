# pages/1_⚙️_Forward_Prediction.py
# Forward prediction: composition + processing -> mechanical properties + alloy family
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys
import plotly.graph_objects as go
from fpdf import FPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import (apply_theme, brand_bar, page_header, section, footer,
                    badge, PLOTLY_LAYOUT, PRIMARY, MUTED)
from alloy_info import estimate_cost, sourcing_suggestions, cost_tier_color

st.set_page_config(page_title="Forward Prediction — AlloyML", page_icon="⚙",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
brand_bar(page_label="Forward Prediction")


# ---- models ---------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        clf = joblib.load(os.path.join(PROJECT_ROOT, "models/classifier_lgbm_pipeline.joblib"))
        reg = joblib.load(os.path.join(PROJECT_ROOT, "models/regressor_lgbm_multi_pipeline.joblib"))
        meta = joblib.load(os.path.join(PROJECT_ROOT, "models/metadata.joblib"))
        return clf, reg, meta
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Run `python src/alloy_ml_research.py` to train.")
        return None, None, None


@st.cache_resource
def load_intervals():
    """Load conformal prediction intervals. Return None gracefully if missing."""
    path = os.path.join(PROJECT_ROOT, "models/prediction_intervals.joblib")
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


page_header("Forward Prediction",
            "Input alloy composition and processing → predict mechanical properties and alloy family.")

clf, reg, meta = load_models()
intervals = load_intervals()
if clf is None:
    st.stop()


# ---- sidebar: inputs ------------------------------------------------------
st.sidebar.header("Composition")
st.sidebar.caption("Element weight percentages")

inputs = {}
element_defaults = [
    ("Al_wt", "Al",  95.0), ("Cu_wt", "Cu",  0.5), ("Fe_wt", "Fe",  0.3),
    ("Mg_wt", "Mg",  1.0),  ("Mn_wt", "Mn",  0.3), ("Si_wt", "Si",  0.6),
    ("Ti_wt", "Ti",  0.1),  ("Zn_wt", "Zn",  0.2), ("Cr_wt", "Cr",  0.1),
    ("Others_wt", "Others", 0.05),
]
sc = st.sidebar.columns(2)
for i, (col, label, default) in enumerate(element_defaults):
    inputs[col] = sc[i % 2].number_input(
        f"{label} (wt%)", min_value=0.0, max_value=100.0,
        value=default, step=0.1, key=f"in_{col}",
    )

st.sidebar.divider()
inputs["Density_g_cm3"] = st.sidebar.number_input(
    "Density (g/cm³)", min_value=0.0, value=2.70, step=0.01,
)

st.sidebar.header("Processing")
alloy_series = st.sidebar.selectbox(
    "Alloy series",
    options=["1xxx","2xxx","3xxx","4xxx","5xxx","6xxx","7xxx","8xxx","Cast"],
    index=5,
)
proc = st.sidebar.selectbox(
    "Processing", options=["Wrought","Cold Worked","Forged","Cast","Die Cast"],
)
temper = st.sidebar.selectbox(
    "Temper",
    options=["O","H12","H14","H16","H18","H34","H38","T1","T3","T4",
             "T5","T6","T651","T73","T76","T81","T87","F"], index=11,
)
form = st.sidebar.selectbox(
    "Form",
    options=["Sheet","Plate","Extrusion","Bar","Rod","Wire","Forging","Casting","Foil","Tube"],
)


# ---- build feature rows (including derived metallurgical features) --------
derived = {
    "Mg_Si_ratio": inputs["Mg_wt"] / (inputs["Si_wt"] + 0.01),
    "Zn_Mg_sum": inputs["Zn_wt"] + inputs["Mg_wt"],
    "Cu_Mg_sum": inputs["Cu_wt"] + inputs["Mg_wt"],
    "total_solute": sum(inputs[c] for c in
                        ["Cu_wt","Mg_wt","Mn_wt","Si_wt","Zn_wt","Cr_wt"]),
}
cls_input = {**inputs, **derived, "Temper": temper, "Form": form, "Processing": proc}
reg_input = {**inputs, **derived, "Series": alloy_series,
             "Temper": temper, "Form": form, "Processing": proc}
cls_df = pd.DataFrame([cls_input])
reg_df = pd.DataFrame([reg_input])


# ---- input summary --------------------------------------------------------
section("Input summary")
total_comp = sum(v for k, v in inputs.items() if k.endswith("_wt"))

s1, s2, s3 = st.columns(3)
s1.metric("Total composition", f"{total_comp:.1f}%",
          delta=f"{total_comp - 100:+.1f}%" if abs(total_comp - 100) > 0.05 else None)
s2.metric("Density", f"{inputs['Density_g_cm3']:.2f} g/cm³")
s3.metric("Target series", alloy_series)

with st.expander("View full composition"):
    comp_df = pd.DataFrame({
        "Element": [c.replace("_wt", "") for c in inputs if c.endswith("_wt")],
        "Weight %": [f"{inputs[c]:.2f}" for c in inputs if c.endswith("_wt")],
    })
    comp_df = comp_df[comp_df["Weight %"].astype(float) > 0].reset_index(drop=True)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ---- predict --------------------------------------------------------------
st.markdown("")  # spacer
if st.button("Predict", use_container_width=True, type="primary"):
    with st.spinner("Running prediction…"):
        try:
            cls_pred = clf.predict(cls_df)[0]
            cls_proba = clf.predict_proba(cls_df)[0]
            reg_pred = reg.predict(reg_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    targets = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
    target_labels = ["UTS (MPa)","YS (MPa)","Elongation (%)","Hardness (HB)","Modulus (GPa)"]
    units = ["MPa","MPa","%","HB","GPa"]
    max_prob = float(cls_proba.max())

    # Predicted family
    section("Predicted alloy family")
    f1, f2 = st.columns([2, 1])
    f1.metric("Predicted series", cls_pred)
    f2.metric("Confidence", f"{max_prob * 100:.1f}%")

    # Top-5 probability chart
    classes = clf.classes_ if hasattr(clf, "classes_") else clf.named_steps["clf"].classes_
    proba_ranked = pd.Series(cls_proba, index=classes).sort_values(ascending=True)[-5:]
    fig = go.Figure(go.Bar(
        y=proba_ranked.index, x=proba_ranked.values * 100,
        orientation="h", marker_color=PRIMARY,
        text=[f"{v*100:.1f}%" for v in proba_ranked.values], textposition="outside",
    ))
    fig.update_layout(height=260, xaxis_title="Probability (%)", **PLOTLY_LAYOUT)
    fig.update_layout(margin=dict(l=70, r=40, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Predicted properties
    section("Predicted mechanical properties")
    ALPHA = 0.90  # default 90% coverage
    if intervals is not None:
        st.caption(
            f"Point estimate with ± {int(ALPHA*100)}% conformal prediction interval "
            "(from 5-fold GroupKFold residuals)."
        )
    cols = st.columns(5)
    target_keys = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
    for col, label, value, unit, tkey in zip(cols, target_labels, reg_pred, units, target_keys):
        if unit == "%":
            disp = f"{value:.1f}%"
        elif unit in ("MPa","HB"):
            disp = f"{value:.0f} {unit}"
        else:
            disp = f"{value:.1f} {unit}"
        delta_txt = None
        if intervals is not None:
            q = intervals["quantiles"][tkey].get(ALPHA)
            if q is not None:
                if unit == "%":
                    delta_txt = f"± {q:.1f}%"
                elif unit in ("MPa","HB"):
                    delta_txt = f"± {q:.0f} {unit}"
                else:
                    delta_txt = f"± {q:.1f} {unit}"
        col.metric(label, disp, delta=delta_txt, delta_color="off")

    # Quick-read character badges
    uts, ys, elong, hb, mod = reg_pred
    chips = []
    if uts >= 450:   chips.append(badge("High strength",   "success"))
    elif uts <= 180: chips.append(badge("Low strength",    "warn"))
    else:            chips.append(badge("Moderate strength","primary"))
    if elong >= 15:  chips.append(badge("Ductile",         "success"))
    elif elong <= 5: chips.append(badge("Brittle",         "warn"))
    if hb >= 120:    chips.append(badge("Hard",            "primary"))
    if mod >= 75:    chips.append(badge("Stiff",           "primary"))
    if ys / uts >= 0.85 and uts >= 300:
        chips.append(badge("Peak-aged temper", "muted"))
    st.markdown(
        '<div style="margin:0.6rem 0 0.2rem 0; display:flex; gap:0.4rem; flex-wrap:wrap;">'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )

    # ---- cost + sourcing --------------------------------------------------
    section("Cost & sourcing (indicative)")
    cost = estimate_cost(cls_pred, form)
    sources = sourcing_suggestions(cls_pred, form)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Unit cost", f"${cost['unit_cost_usd_kg']:.2f} / kg",
              delta=f"${cost['low']:.2f}–${cost['high']:.2f} range",
              delta_color="off")
    c2.markdown("**Price tier**")
    c2.markdown(badge(cost["tier"].title(), cost_tier_color(cost["tier"])),
                unsafe_allow_html=True)
    c2.caption(f"Form premium ×{cost['form_premium']:.2f}")
    with c3:
        st.markdown("**Where to source**")
        st.markdown("\n".join(f"- {s}" for s in sources))
    st.caption(
        f"*{cost['note']}* — indicative 2025-2026 prices. "
        "Request a live quote for procurement."
    )

    with st.expander("Details"):
        props_table = pd.DataFrame({
            "Property": target_labels,
            "Predicted value": [f"{v:.2f}" for v in reg_pred],
            "Unit": units,
        })
        st.dataframe(props_table, use_container_width=True, hide_index=True)

    # ---- PDF export -------------------------------------------------------
    section("Export")
    st.caption("Download a formatted material data sheet with these predictions.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 15)
    pdf.cell(0, 10, "Alloy Material Data Sheet", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, 22, 200, 22)
    pdf.ln(6)

    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 7, "1. Chemical composition (wt%)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    comp_text = "   ".join(f"{c.replace('_wt','')}: {inputs[c]}%"
                           for c in inputs if c.endswith("_wt") and inputs[c] > 0)
    pdf.multi_cell(0, 5.5, comp_text)
    pdf.cell(0, 5.5, f"Density: {inputs['Density_g_cm3']:.2f} g/cm3",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 7, "2. Processing parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    for line in [f"Target series: {alloy_series}", f"Processing: {proc}",
                 f"Temper: {temper}", f"Form: {form}"]:
        pdf.cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 7, "3. Predicted mechanical properties", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(70, 6, "Property", border=1)
    pdf.cell(60, 6, "Predicted", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(70, 6, "Predicted family", border=1)
    pdf.cell(60, 6, f"{cls_pred} ({max_prob*100:.1f}%)", border=1,
             new_x="LMARGIN", new_y="NEXT")
    for label, val, unit in zip(target_labels, reg_pred, units):
        pdf.cell(70, 6, label, border=1)
        pdf.cell(60, 6, f"{val:.2f} {unit}", border=1,
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "Generated by AlloyML - IIT Roorkee", align="C")

    st.download_button(
        "Download data sheet (PDF)",
        data=bytes(pdf.output()),   # fpdf2 returns bytearray; Streamlit wants bytes
        file_name="alloy_data_sheet.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

footer()
