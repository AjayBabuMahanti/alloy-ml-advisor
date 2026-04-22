# app.py — Home landing page
import streamlit as st
from styles import apply_theme, brand_bar, page_header, section, stat_band, nav_card, footer

st.set_page_config(
    page_title="AlloyML",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
brand_bar(page_label="Home")

page_header(
    "Predict, compose, and select aluminum alloys.",
    "AlloyML is an AI-assisted advisor built on 783 real aluminum alloy records from "
    "ASM Handbook, Light Alloys Databook, and Davis Aluminum Specialty. "
    "Predict mechanical properties, search for matches, generate new compositions, "
    "and consult a materials-engineering LLM — all in one app.",
)

# ---- hero stats band ------------------------------------------------------
stat_band([
    ("Records",       "783",       "Real, handbook-sourced"),
    ("Regressor R²",  "0.85 avg",  "5 property targets"),
    ("90% PI (UTS)",  "± 113 MPa", "Conformal, 5-fold CV"),
    ("Case studies",  "3",         "Aerospace · Auto · Marine"),
])

# ---- navigation grid ------------------------------------------------------
section("Tools")

pages = [
    {
        "icon": "⚙",
        "title": "Forward Prediction",
        "body": "Enter composition and processing to predict mechanical properties (with 90% confidence intervals) and the alloy series.",
        "target": "pages/1_⚙️_Forward_Prediction.py",
        "key": "fwd",
    },
    {
        "icon": "⇄",
        "title": "Inverse Prediction",
        "body": "Specify target properties. Find the closest real alloy, or generate a candidate composition via differential evolution.",
        "target": "pages/2_🔍_Inverse_Prediction.py",
        "key": "inv",
    },
    {
        "icon": "▤",
        "title": "Batch Prediction",
        "body": "Upload a CSV of alloys to get predictions, intervals, and cost estimates for every row at once.",
        "target": "pages/3_📁_Batch_Prediction.py",
        "key": "batch",
    },
    {
        "icon": "✦",
        "title": "AI Assistant",
        "body": "Chat with a materials-engineering LLM (Mistral) for alloy recommendations, tradeoffs, and application guidance.",
        "target": "pages/4_🤖_AI_Assistant.py",
        "key": "chat",
    },
    {
        "icon": "◎",
        "title": "Case Studies",
        "body": "Three real components (aerospace spar, cast wheel, marine hull) — see how AlloyML's recommendations compare to industry choices.",
        "target": "pages/5_📚_Case_Studies.py",
        "key": "cases",
    },
]

# 5 tiles → 3-column top row + 2-column bottom row (or 2x2+1)
row1 = st.columns(3, gap="medium")
row2 = st.columns(3, gap="medium")
slots = row1 + row2
for col, p in zip(slots, pages):
    with col:
        st.markdown(nav_card(p["title"], p["body"], icon=p["icon"]),
                    unsafe_allow_html=True)
        if st.button(f"Open  →", key=p["key"], use_container_width=True):
            st.switch_page(p["target"])

# ---- supporting info ------------------------------------------------------
section("How it works")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("**1. Curated dataset**")
    st.caption(
        "Real alloy compositions and mechanical properties pulled from three standard "
        "metallurgy references, extended with physics-informed augmentation."
    )
with c2:
    st.markdown("**2. Tuned models**")
    st.caption(
        "LightGBM regressor and RandomForest classifier, tuned via RandomizedSearchCV "
        "with 4 physics-informed derived features (Mg/Si, Zn+Mg, Cu+Mg, total solute)."
    )
with c3:
    st.markdown("**3. Multimodal use**")
    st.caption(
        "Forward and inverse prediction, batch CSV scoring, and a conversational assistant "
        "grounded in the dataset — everything an engineer needs to pick an alloy."
    )

with st.expander("Team and acknowledgements"):
    st.markdown(
        "- **Advisor:** Prof. Anuj Bisht (Dept. of Mechanical & Industrial Engineering)  \n"
        "- **Team:** Akshat Jain, Alumolu Rakesh Reddy, Mahanti Ajay Babu (22117079), Himanshu Panwar, Ashok Godara  \n"
        "- **Evaluation board:** Production & Industrial  \n"
        "- **Data sources:** ASM Handbook Vol. 2 · Hussey & Wilson *Light Alloys Directory and Databook* · Davis *Aluminum and Aluminum Alloys* (ASM Specialty)"
    )

footer()
