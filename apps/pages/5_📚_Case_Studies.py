# pages/5_📚_Case_Studies.py
# Industrial case studies: compare AlloyML's recommendation to the real-world
# alloy actually used in each component.
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import (apply_theme, brand_bar, page_header, section, footer,
                    badge, PLOTLY_LAYOUT, PRIMARY, SUCCESS, MUTED)
from alloy_info import estimate_cost, cost_tier_color, sourcing_suggestions

st.set_page_config(page_title="Case Studies — AlloyML", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
brand_bar(page_label="Case Studies")


# ============================================================================
# CASE STUDIES
# Target properties below are representative specifications compiled from
# published handbook data and industry references. Each one is solved by
# (a) passing the target through a nearest-neighbor search on the dataset and
# (b) comparing the #1 recommendation against the alloy actually used in
# industry for that component.
# ============================================================================
CASES = [
    {
        "id": "aero",
        "title": "Aerospace wing spar",
        "sector": "Aerospace",
        "description": (
            "Upper-wing spar caps and fuselage frame members on commercial transport "
            "aircraft (e.g. Boeing 737, Airbus A320) need the highest strength-to-weight "
            "ratio achievable in aluminum, combined with adequate fracture toughness."
        ),
        "requirements": {
            "UTS_MPa": 570,
            "YS_MPa": 500,
            "Elongation_pct": 11,
            "Hardness_HB": 150,
            "Modulus_GPa": 72,
        },
        "industry_choice": "7075-T6",
        "industry_rationale": (
            "7075-T6 is the canonical aerospace plate/extrusion alloy. "
            "Al-Zn-Mg-Cu precipitation strengthening via η' (MgZn₂) phase gives peak "
            "strength after T6 aging. Trade-off: poor weldability and stress-corrosion "
            "susceptibility, mitigated by T73/T76 overaging on fatigue-critical parts."
        ),
        "form": "Plate",
        "image_concept": "High-strength 7075 alloy plate, machined wing-box structure",
    },
    {
        "id": "auto",
        "title": "Automotive cast wheel",
        "sector": "Automotive",
        "description": (
            "Light-alloy wheels on passenger vehicles are produced almost exclusively by "
            "low-pressure die casting or forging. Cast wheels must balance strength, "
            "ductility for impact tolerance, and excellent castability for complex shapes."
        ),
        "requirements": {
            "UTS_MPa": 265,
            "YS_MPa": 200,
            "Elongation_pct": 7,
            "Hardness_HB": 82,
            "Modulus_GPa": 72,
        },
        "industry_choice": "A356-T6",
        "industry_rationale": (
            "A356.0-T6 (AlSi7Mg0.3) dominates light-alloy wheel production. The 7% Si "
            "gives excellent castability and fluidity, and T6 heat treatment precipitates "
            "Mg₂Si for strength. Outstanding fatigue life and dimensional stability; "
            "widely produced by low-pressure and gravity casting."
        ),
        "form": "Casting",
        "image_concept": "Low-pressure cast A356 wheel, machined and polished",
    },
    {
        "id": "marine",
        "title": "Marine hull plate",
        "sector": "Marine",
        "description": (
            "Hull plates on aluminum boats, patrol craft, and fast ferries are welded "
            "from 5xxx Al-Mg sheet and plate. Required properties: good strength without "
            "heat treatment, excellent seawater corrosion resistance, and high weldability."
        ),
        "requirements": {
            "UTS_MPa": 317,
            "YS_MPa": 228,
            "Elongation_pct": 16,
            "Hardness_HB": 75,
            "Modulus_GPa": 71,
        },
        "industry_choice": "5083-H116",
        "industry_rationale": (
            "5083-H116 (Al-4.4Mg-0.7Mn) is the marine aluminum standard (ASTM B928). "
            "Non-heat-treatable: strengthened by strain hardening (H116 = special temper "
            "for exfoliation resistance). Retains strength in welded joints, resists "
            "seawater pitting and stress corrosion; widely used in hulls and superstructures."
        ),
        "form": "Plate",
        "image_concept": "Welded 5083 aluminum boat hull, shipyard construction",
    },
]


# ---- loaders --------------------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "data/alloy_dataset_final.csv"))


@st.cache_resource
def load_regressor():
    try:
        return joblib.load(os.path.join(PROJECT_ROOT,
                                        "models/regressor_lgbm_multi_pipeline.joblib"))
    except FileNotFoundError:
        return None


@st.cache_resource
def load_intervals():
    try:
        return joblib.load(os.path.join(PROJECT_ROOT, "models/prediction_intervals.joblib"))
    except FileNotFoundError:
        return None


def nearest_match(df: pd.DataFrame, targets_dict: dict, form: str = None,
                  prefer_seed: bool = True):
    """Return (top1_row, top5_df) from the dataset nearest to target properties.

    If prefer_seed is True, augmented variants (names matching `_v\\d+$`) are
    lightly penalized so the top match is a real seed alloy whenever possible.
    """
    keys = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
    target_vec = np.array([targets_dict[k] for k in keys]).reshape(1, -1)
    data_props = df[keys].values
    ranges = data_props.max(axis=0) - data_props.min(axis=0)
    ranges[ranges == 0] = 1
    distances = euclidean_distances(
        (target_vec - data_props.min(axis=0)) / ranges,
        (data_props - data_props.min(axis=0)) / ranges,
    )[0]
    if form is not None:
        form_mask = df["Form"].values == form
        if form_mask.sum() >= 5:
            distances[~form_mask] = 1e10
    if prefer_seed:
        is_aug = df["Alloy"].str.contains(r"_v\d+$", regex=True).values
        distances = distances + is_aug * 0.02   # small tie-breaker penalty
    order = np.argsort(distances)
    top5 = df.iloc[order[:5]].copy()
    top5["Distance"] = distances[order[:5]]
    top5["Match %"] = ((1 / (1 + top5["Distance"])) * 100).round(1)
    # Strip _vN suffix for display (augmented variants become their seed name)
    top5["Alloy"] = top5["Alloy"].str.replace(r"_v\d+$", "", regex=True)
    return top5.iloc[0], top5


def agreement_verdict(recommended_alloy: str, industry_alloy: str) -> tuple:
    """Return (verdict_text, badge_kind)."""
    # Normalize: extract base alloy like "7075" and series like "7xxx"
    def base(s):
        s = str(s).upper().replace(".0","")
        # Grab first 4 characters that include digits
        import re
        m = re.match(r"([A-Z]?\d+)", s)
        return m.group(1) if m else s
    def series(s):
        b = base(s)
        # Heuristic: 4-digit wrought or 3-digit cast
        if b.isdigit() and len(b) == 4:
            return b[0] + "xxx"
        return "Cast"
    r_base, i_base = base(recommended_alloy), base(industry_alloy)
    r_ser,  i_ser  = series(recommended_alloy), series(industry_alloy)
    if r_base == i_base:
        return ("Model recommended the exact industry alloy.", "success")
    if r_ser == i_ser:
        return (f"Model recommended a {r_ser} alloy — same series as industry choice.", "success")
    return ("Model recommended a different series than industry practice — worth investigating.", "warn")


# ---- page -----------------------------------------------------------------
page_header(
    "Industrial Case Studies",
    "Three real engineering components. For each one, AlloyML is given only the target "
    "mechanical properties, and its recommendation is compared to the alloy actually "
    "used in that component by industry. This is a practical validation of the tool "
    "on decisions engineers make every day.",
)

df = load_dataset()
reg = load_regressor()
intervals = load_intervals()

# tab interface so each case has its own space without scrolling conflicts
tabs = st.tabs([c["title"] for c in CASES])

for tab, case in zip(tabs, CASES):
    with tab:
        # ---- Context band -------------------------------------------------
        st.markdown(f"**Sector:** {case['sector']}  ·  **Form:** {case['form']}")
        st.markdown(case["description"])

        section("Target mechanical properties")
        tgt_cols = st.columns(5)
        labels_units = [
            ("UTS_MPa",        "UTS",        "MPa"),
            ("YS_MPa",         "Yield",      "MPa"),
            ("Elongation_pct", "Elongation", "%"),
            ("Hardness_HB",    "Hardness",   "HB"),
            ("Modulus_GPa",    "Modulus",    "GPa"),
        ]
        for col, (key, lbl, unit) in zip(tgt_cols, labels_units):
            v = case["requirements"][key]
            if unit == "%":
                col.metric(lbl, f"{v:.0f}%")
            elif unit in ("MPa","HB"):
                col.metric(lbl, f"{v:.0f} {unit}")
            else:
                col.metric(lbl, f"{v:.1f} {unit}")

        # ---- Nearest match ------------------------------------------------
        top1, top5 = nearest_match(df, case["requirements"], form=case["form"])
        verdict, v_kind = agreement_verdict(top1["Alloy"], case["industry_choice"])

        section("AlloyML recommendation vs industry choice")
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            st.markdown(f"#### AlloyML says: `{top1['Alloy']}` · {top1['Temper']} · {top1['Form']}")
            st.markdown(f"Match quality: **{top5.iloc[0]['Match %']:.1f}%** (normalized distance "
                        f"{top5.iloc[0]['Distance']:.3f})")
            # show its key properties
            st.markdown(
                f"- UTS: **{top1['UTS_MPa']:.0f} MPa** (target {case['requirements']['UTS_MPa']:.0f})  \n"
                f"- YS: **{top1['YS_MPa']:.0f} MPa** (target {case['requirements']['YS_MPa']:.0f})  \n"
                f"- Elongation: **{top1['Elongation_pct']:.1f}%** (target {case['requirements']['Elongation_pct']:.0f}%)  \n"
                f"- Hardness: **{top1['Hardness_HB']:.0f} HB** (target {case['requirements']['Hardness_HB']:.0f})  \n"
                f"- Modulus: **{top1['Modulus_GPa']:.1f} GPa** (target {case['requirements']['Modulus_GPa']:.1f})"
            )

        with a2:
            st.markdown(f"#### Industry uses: `{case['industry_choice']}`")
            st.caption(case["industry_rationale"])

        st.markdown(badge(verdict, v_kind), unsafe_allow_html=True)

        # ---- Side-by-side property chart ----------------------------------
        keys = ["UTS_MPa","YS_MPa","Elongation_pct","Hardness_HB","Modulus_GPa"]
        labels = ["UTS","Yield","Elong","Hard","Mod"]
        fig = go.Figure(data=[
            go.Bar(name="Target",     x=labels,
                   y=[case["requirements"][k] for k in keys],
                   marker_color=PRIMARY),
            go.Bar(name=f"AlloyML → {top1['Alloy']}", x=labels,
                   y=[float(top1[k]) for k in keys],
                   marker_color=SUCCESS),
        ])
        fig.update_layout(
            barmode="group", height=360,
            title=f"Target vs AlloyML recommendation — {case['title']}",
            hovermode="x unified", **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Top-5 matches table ------------------------------------------
        section("Top 5 nearest alloys in the dataset")
        tbl = top5[["Alloy","Series","Temper","Form","UTS_MPa","YS_MPa",
                    "Elongation_pct","Hardness_HB","Modulus_GPa","Match %"]].copy()
        tbl = tbl.rename(columns={
            "UTS_MPa":"UTS (MPa)", "YS_MPa":"YS (MPa)",
            "Elongation_pct":"Elong (%)", "Hardness_HB":"Hardness (HB)",
            "Modulus_GPa":"Modulus (GPa)",
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ---- Cost comparison ----------------------------------------------
        section("Cost comparison")
        rec_cost = estimate_cost(top1["Series"], top1["Form"])

        # Try to find the industry choice in the dataset to get its series/form
        ind_series = "7xxx" if "7" in case["industry_choice"][:1] else (
                     "Cast" if "A356" in case["industry_choice"] or "356" in case["industry_choice"]
                     else "5xxx" if "5" in case["industry_choice"][:1] else "6xxx")
        ind_cost = estimate_cost(ind_series, case["form"])

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("AlloyML rec unit cost", f"${rec_cost['unit_cost_usd_kg']:.2f}/kg",
                   delta=f"{rec_cost['tier']}", delta_color="off")
        cc2.metric("Industry choice unit cost", f"${ind_cost['unit_cost_usd_kg']:.2f}/kg",
                   delta=f"{ind_cost['tier']}", delta_color="off")
        diff = rec_cost['unit_cost_usd_kg'] - ind_cost['unit_cost_usd_kg']
        cc3.metric("Delta", f"${diff:+.2f}/kg",
                   delta="rec cheaper" if diff < -0.1 else
                         ("rec pricier" if diff > 0.1 else "similar"),
                   delta_color="off")

        st.caption(f"*{rec_cost['note']}*")

        # ---- Sourcing -----------------------------------------------------
        section("Where to source this alloy")
        sources = sourcing_suggestions(top1["Series"], top1["Form"])
        st.markdown("\n".join(f"- {s}" for s in sources))

footer()
