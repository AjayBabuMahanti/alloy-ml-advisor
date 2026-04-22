# pages/2_🔍_Inverse_Prediction.py
# Inverse prediction: target properties -> matching alloys OR generated composition
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings
import joblib
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import differential_evolution
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import (apply_theme, brand_bar, page_header, section, footer,
                    badge, PLOTLY_LAYOUT, PRIMARY, SUCCESS)
from alloy_info import estimate_cost, cost_tier_color

st.set_page_config(page_title="Inverse Prediction — AlloyML", page_icon="⇄",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
brand_bar(page_label="Inverse Prediction")


# ---- loaders --------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv(os.path.join(PROJECT_ROOT, "data/alloy_dataset_final.csv"))
    except FileNotFoundError:
        return pd.read_csv(os.path.join(PROJECT_ROOT, "data/alloy_dataset_v2.csv"))


@st.cache_resource
def load_regressor():
    try:
        return joblib.load(os.path.join(PROJECT_ROOT,
                                        "models/regressor_lgbm_multi_pipeline.joblib"))
    except FileNotFoundError:
        return None


def generate_composition(reg_model, targets_dict, series, temper, form, processing):
    t_list = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
    target_vec = np.array([targets_dict[t] for t in t_list])
    target_ranges = np.array([600.0, 500.0, 40.0, 150.0, 15.0])

    bounds = [
        (80.0, 99.9), (0.0, 7.0), (0.0, 2.0), (0.0, 8.0), (0.0, 2.0),
        (0.0, 14.0), (0.0, 0.3), (0.0, 10.0), (0.0, 0.5), (0.0, 3.0),
        (2.55, 2.90),
    ]

    def objective(x):
        Al, Cu, Fe, Mg, Mn, Si, Ti, Zn, Cr, Others, Dens = x
        comp_sum = Al + Cu + Fe + Mg + Mn + Si + Ti + Zn + Cr + Others
        penalty = 20.0 * abs(comp_sum - 100.0)
        row = {
            "Al_wt": Al, "Cu_wt": Cu, "Fe_wt": Fe, "Mg_wt": Mg, "Mn_wt": Mn,
            "Si_wt": Si, "Ti_wt": Ti, "Zn_wt": Zn, "Cr_wt": Cr, "Others_wt": Others,
            "Density_g_cm3": Dens,
            "Mg_Si_ratio": Mg / (Si + 0.01),
            "Zn_Mg_sum": Zn + Mg, "Cu_Mg_sum": Cu + Mg,
            "total_solute": Cu + Mg + Mn + Si + Zn + Cr,
            "Series": series, "Temper": temper, "Form": form, "Processing": processing,
        }
        try:
            pred = reg_model.predict(pd.DataFrame([row]))[0]
            return float(np.mean(((pred - target_vec) / target_ranges) ** 2)) + penalty
        except Exception:
            return 1e6

    return differential_evolution(objective, bounds, seed=42, maxiter=25,
                                  popsize=10, tol=1e-3, polish=True, disp=False)


# ---- UI -------------------------------------------------------------------
page_header("Inverse Prediction",
            "Specify target mechanical properties to find matching real alloys, or generate a new candidate composition.")

df = load_data()
if df is None:
    st.stop()

targets = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
target_labels = ["UTS (MPa)","YS (MPa)","Elongation (%)","Hardness (HB)","Modulus (GPa)"]

# Sidebar: target properties + constraints
st.sidebar.header("Target mechanical properties")
desired = {}
sc = st.sidebar.columns(2)
for i, (t, lbl) in enumerate(zip(targets, target_labels)):
    mid = float((df[t].min() + df[t].max()) / 2)
    step = 5.0 if "MPa" in lbl or "HB" in lbl else (0.5 if "GPa" in lbl else 1.0)
    desired[t] = sc[i % 2].number_input(
        lbl, min_value=0.0, max_value=float(df[t].max() * 1.5),
        value=mid, step=step, help=f"Dataset range: {df[t].min():.0f}-{df[t].max():.0f}",
    )

st.sidebar.header("Constraints (optional)")
preferred_series = st.sidebar.selectbox(
    "Preferred alloy series", ["Any"] + sorted(df["Series"].unique().tolist()),
)
preferred_form = st.sidebar.selectbox(
    "Preferred form", ["Any"] + sorted(df["Form"].unique().tolist()),
)

# Target summary
section("Target summary")
m = st.columns(5)
m[0].metric("UTS",        f"{desired['UTS_MPa']:.0f} MPa")
m[1].metric("YS",         f"{desired['YS_MPa']:.0f} MPa")
m[2].metric("Elongation", f"{desired['Elongation_pct']:.1f}%")
m[3].metric("Hardness",   f"{desired['Hardness_HB']:.0f} HB")
m[4].metric("Modulus",    f"{desired['Modulus_GPa']:.1f} GPa")


# ==========================================================================
# Mode 1: find nearest from dataset
# ==========================================================================
st.markdown("")
section("Mode 1 · Find matching alloys (from database)")
st.caption("Normalized-distance nearest-neighbor search over 783 real alloy records.")

if st.button("Find matching alloys", use_container_width=True, type="primary"):
    with st.spinner("Searching…"):
        target_vec = np.array(list(desired.values())).reshape(1, -1)
        data_props = df[targets].values
        ranges = data_props.max(axis=0) - data_props.min(axis=0)
        ranges[ranges == 0] = 1
        distances = euclidean_distances(
            (target_vec - data_props.min(axis=0)) / ranges,
            (data_props - data_props.min(axis=0)) / ranges,
        )[0]

        mask = np.ones(len(df), dtype=bool)
        if preferred_series != "Any": mask &= df["Series"].values == preferred_series
        if preferred_form   != "Any": mask &= df["Form"].values   == preferred_form
        if mask.sum() == 0:
            st.warning("No alloys match the filter. Showing unfiltered results.")
            mask[:] = True

        fd = distances.copy(); fd[~mask] = 1e10
        top_idx = np.argsort(fd)[:min(10, int(mask.sum()))]
        top = df.iloc[top_idx].copy()
        top["Match %"] = ((1 / (1 + distances[top_idx])) * 100).round(1)

    # Add cost column to the table
    top["Cost USD/kg"] = [
        f"${estimate_cost(s, f)['unit_cost_usd_kg']:.2f}"
        for s, f in zip(top["Series"], top["Form"])
    ]

    st.markdown("**Top 10 matches**")
    display_df = top[["Alloy","Series","Form","Processing","Temper"]
                     + targets + ["Match %", "Cost USD/kg"]]
    display_df = display_df.rename(columns={t: l for t, l in zip(targets, target_labels)})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best = df.iloc[top_idx[0]]
    st.markdown(f"**Best match:** `{best['Alloy']}`  ·  "
                f"Series {best['Series']}  ·  Form {best['Form']}  ·  "
                f"Temper {best['Temper']}  ·  Match {top.iloc[0]['Match %']:.1f}%")

    c1, c2 = st.columns([1, 1])
    with c1:
        comp_cols = ["Al_wt","Cu_wt","Fe_wt","Mg_wt","Mn_wt",
                     "Si_wt","Ti_wt","Zn_wt","Cr_wt","Others_wt"]
        comp_disp = pd.DataFrame([
            {"Element": c.replace("_wt", ""), "Weight %": f"{best[c]:.2f}"}
            for c in comp_cols if best[c] > 0.01
        ])
        st.markdown("**Best-match composition**")
        st.dataframe(comp_disp, use_container_width=True, hide_index=True)

    with c2:
        fig = go.Figure(data=[
            go.Bar(name="Target",     x=target_labels,
                   y=[desired[t] for t in targets], marker_color=PRIMARY),
            go.Bar(name="Best match", x=target_labels,
                   y=[best[t] for t in targets], marker_color=SUCCESS),
        ])
        fig.update_layout(barmode="group", height=340, title="Target vs best match",
                          hovermode="x unified", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# Mode 2: generate optimal composition via differential evolution
# ==========================================================================
st.markdown("")
section("Mode 2 · Generate candidate composition (optimization)")
st.caption("Uses scipy differential evolution with the forward model as the objective. "
           "Takes ~30–60 seconds per run.")

g1, g2, g3, g4 = st.columns(4)
with g1:
    gen_series = st.selectbox(
        "Target series",
        options=["1xxx","2xxx","3xxx","4xxx","5xxx","6xxx","7xxx","8xxx","Cast"],
        index=5, key="gen_series",
    )
with g2:
    gen_temper = st.selectbox(
        "Temper",
        options=["O","H12","H14","H16","H18","T3","T4","T5","T6","T651","T73","T76","T81","F"],
        index=8, key="gen_temper",
    )
with g3:
    gen_form = st.selectbox(
        "Form",
        options=["Sheet","Plate","Extrusion","Bar","Rod","Wire","Forging","Casting","Foil","Tube"],
        index=0, key="gen_form",
    )
with g4:
    gen_proc = st.selectbox(
        "Processing",
        options=["Wrought","Cold Worked","Forged","Cast","Die Cast"],
        key="gen_proc",
    )

if st.button("Generate optimal composition", use_container_width=True):
    reg = load_regressor()
    if reg is None:
        st.error("Regressor model not found. Train it first with "
                 "`python src/alloy_ml_research.py`.")
    else:
        with st.spinner("Running differential evolution… (this takes ~30-60s)"):
            result = generate_composition(reg, desired, gen_series,
                                          gen_temper, gen_form, gen_proc)

        # Unpack + normalize to exactly 100%
        Al, Cu, Fe, Mg, Mn, Si, Ti, Zn, Cr, Others, Dens = result.x
        comp_sum = Al + Cu + Fe + Mg + Mn + Si + Ti + Zn + Cr + Others
        scale = 100.0 / comp_sum
        norm = dict(zip(
            ["Al","Cu","Fe","Mg","Mn","Si","Ti","Zn","Cr","Others"],
            [v * scale for v in [Al, Cu, Fe, Mg, Mn, Si, Ti, Zn, Cr, Others]],
        ))

        row = {f"{k}_wt": v for k, v in norm.items()}
        row.update({
            "Density_g_cm3": Dens,
            "Mg_Si_ratio": norm["Mg"] / (norm["Si"] + 0.01),
            "Zn_Mg_sum": norm["Zn"] + norm["Mg"],
            "Cu_Mg_sum": norm["Cu"] + norm["Mg"],
            "total_solute": sum(norm[k] for k in ["Cu","Mg","Mn","Si","Zn","Cr"]),
            "Series": gen_series, "Temper": gen_temper,
            "Form": gen_form, "Processing": gen_proc,
        })
        predicted = reg.predict(pd.DataFrame([row]))[0]

        st.success(f"Converged in {result.nit} iterations (objective = {result.fun:.4f}).")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Generated composition**")
            comp_disp = pd.DataFrame([
                {"Element": k, "Weight %": f"{v:.2f}"}
                for k, v in norm.items() if v > 0.01
            ])
            st.dataframe(comp_disp, use_container_width=True, hide_index=True)
            st.markdown(f"**Density:** {Dens:.3f} g/cm³  \n"
                        f"**Processing:** {gen_series} · {gen_temper} · {gen_form} · {gen_proc}")

        with c2:
            st.markdown("**Predicted vs target**")
            cmp_df = pd.DataFrame({
                "Property": target_labels,
                "Target":   [f"{desired[t]:.1f}" for t in targets],
                "Predicted":[f"{v:.1f}" for v in predicted],
                "|err|":    [f"{abs(v - desired[t]):.1f}" for v, t in zip(predicted, targets)],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        fig = go.Figure(data=[
            go.Bar(name="Target",              x=target_labels,
                   y=[desired[t] for t in targets], marker_color=PRIMARY),
            go.Bar(name="Generated",           x=target_labels,
                   y=[float(v) for v in predicted], marker_color=SUCCESS),
        ])
        fig.update_layout(barmode="group", height=340,
                          title="Target vs generated-composition prediction",
                          hovermode="x unified", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # ---- cost estimate for the generated composition's series+form ----
        cost = estimate_cost(gen_series, gen_form)
        k1, k2, k3 = st.columns([1, 1, 1])
        k1.metric("Indicative cost", f"${cost['unit_cost_usd_kg']:.2f} / kg",
                  delta=f"${cost['low']:.2f}–${cost['high']:.2f}", delta_color="off")
        k2.markdown("**Price tier**")
        k2.markdown(badge(cost["tier"].title(), cost_tier_color(cost["tier"])),
                    unsafe_allow_html=True)
        k3.markdown("**Form premium**")
        k3.markdown(f"×{cost['form_premium']:.2f} (form: {gen_form})")

        st.info("The generated composition is a model recommendation. "
                "Cross-check against handbook values and the nearest real alloy (Mode 1).")

footer()
