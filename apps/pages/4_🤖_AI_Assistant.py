# pages/4_🤖_AI_Assistant.py
# LLM chatbot for alloy selection guidance (Mistral)
import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_theme, brand_bar, page_header, footer

st.set_page_config(page_title="AI Assistant — AlloyML", page_icon="✦",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()
brand_bar(page_label="AI Assistant")


# ---- loaders --------------------------------------------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(os.path.join(PROJECT_ROOT, "data/alloy_dataset_final.csv"))
    except FileNotFoundError:
        return None


@st.cache_resource
def get_mistral_client():
    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        st.error("`mistralai` not installed. Run `pip install mistralai` in the venv.")
        return None, None
    try:
        api_key = st.secrets["mistral"]["api_key"]
        model = st.secrets["mistral"].get("model", "ministral-8b-latest")
    except Exception:
        st.error("Mistral API key not found in `.streamlit/secrets.toml`.")
        st.code(
            '[mistral]\napi_key = "YOUR_KEY"\nmodel   = "ministral-8b-latest"',
            language="toml",
        )
        return None, None
    return Mistral(api_key=api_key), model


def build_dataset_context(df: pd.DataFrame) -> str:
    if df is None:
        return "(dataset unavailable)"
    series_counts = df.groupby("Series").agg(
        n_records=("Alloy", "count"),
        n_unique=("Alloy", "nunique"),
        uts_min=("UTS_MPa", "min"),
        uts_max=("UTS_MPa", "max"),
    ).round(1)
    series_char = {
        "1xxx": "Commercially pure Al (>99%). Excellent corrosion/conductivity; very low strength. Food, electrical, chemical.",
        "2xxx": "Al-Cu, heat-treatable. High strength, moderate corrosion. Aerospace structure (2024, 2014).",
        "3xxx": "Al-Mn. Moderate strength, excellent formability. Cookware, beverage cans, general sheet (3003, 3004).",
        "4xxx": "Al-Si. Used mainly as welding filler (4043) and piston forgings (4032).",
        "5xxx": "Al-Mg, non-heat-treatable. Excellent marine corrosion resistance + weldability (5052, 5083).",
        "6xxx": "Al-Mg-Si, heat-treatable. Most versatile: good strength + corrosion + extrudability (6061, 6063, 6082).",
        "7xxx": "Al-Zn-Mg(-Cu). Highest-strength aluminum alloys. Aerospace wing/fuselage (7075, 7050).",
        "8xxx": "Misc (Al-Li, Al-Fe). Al-Li (8090) gives low density + good stiffness for aerospace.",
        "Cast": "Cast alloys (A356, A380, etc.). Complex shapes; Si improves castability.",
    }
    lines = [f"Dataset: {len(df)} records, {df['Alloy'].nunique()} unique alloy designations.",
             "Series overview (with UTS ranges in this dataset):"]
    for s in sorted(series_counts.index):
        c = series_counts.loc[s]
        lines.append(
            f"- {s}: {int(c['n_records'])} records, {int(c['n_unique'])} alloys, "
            f"UTS {c['uts_min']:.0f}-{c['uts_max']:.0f} MPa. {series_char.get(s, '')}"
        )
    notable = df.groupby("Alloy").size().sort_values(ascending=False).head(15).index.tolist()
    lines.append("")
    lines.append("Notable alloys present: " + ", ".join(notable) + ".")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are the AlloyML assistant: an expert materials engineer who helps users select aluminum alloys.

RULES:
- Recommend SPECIFIC alloy designations (e.g., "6061-T6", "7075-T6"), not vague categories.
- Always state the tradeoffs honestly (if 7075 is strongest but has poor weldability/corrosion, say so).
- Keep responses concise and practical. Prefer short paragraphs and bullet points.
- If the application is unclear, ask ONE focused clarifying question.
- Only make claims backed by standard metallurgy or the dataset below.
- Never invent alloys that aren't standard AA designations.
- When relevant, mention rough cost tiers using the reference below, and point users
  to common distributors (Online Metals, Metal Supermarkets, McMaster-Carr for small
  quantities; Alcoa, Kaiser, Constellium, Hindalco / NALCO / Jindal in India for mill
  quantities). Be explicit that the numbers are indicative, not live quotes.

INDICATIVE PRICE RANGES (USD/kg, 2025-2026, wrought product):
- 1xxx: $2.20-$3.20 (pure Al, foil & electrical)
- 2xxx: $5.80-$9.50 (aerospace grade; Cu adds cost)
- 3xxx: $2.60-$3.80 (can/cookware)
- 4xxx: $3.80-$5.50 (welding filler, pistons)
- 5xxx: $3.00-$4.80 (marine grade)
- 6xxx: $2.80-$4.20 (most common structural)
- 7xxx: $6.50-$11.00 (premium aerospace)
- 8xxx: $10-$18 (Al-Li, most expensive)
- Cast:  $3.20-$5.50
Form premiums: forging ×1.4, wire ×1.2, extrusion ×1.15, plate ×1.1, sheet ×1.0, casting ×0.95.

DATASET CONTEXT:
{dataset_context}

APP CONTEXT:
- Forward Prediction page: composition + processing -> predicted mechanical properties
  (now with 90% conformal prediction intervals and cost/sourcing info).
- Inverse Prediction page: target properties -> find matching alloys or generate a composition.
- Batch Prediction page: bulk CSV predictions with intervals and cost columns.
- Case Studies page: three industrial components (aerospace wing spar, cast wheel,
  marine hull plate) where AlloyML's pick is compared against the real industry choice.
- This page (AI Assistant): conversational guidance on alloy selection.
"""


# ---- UI -------------------------------------------------------------------
page_header("AI Assistant",
            "Chat with a materials-engineering LLM about alloy selection, properties, and applications.")

df = load_dataset()
client, model = get_mistral_client()
if client is None:
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: quick prompts + reset
st.sidebar.header("Example questions")
example_qs = [
    "I need a lightweight alloy for a bicycle frame. What do you recommend?",
    "Compare 6061-T6 and 7075-T6 for an aerospace bracket.",
    "Which aluminum alloy is best for marine applications?",
    "I need ~350 MPa UTS with good weldability. What are my options?",
    "What is the difference between T6 and T73 heat treatments?",
    "Why is 5083 commonly used in boat hulls?",
]
for q in example_qs:
    if st.sidebar.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.caption(f"Model: `{model}`")
st.sidebar.caption(f"Records available: {len(df)}" if df is not None else "Dataset unavailable")


# Empty state
if not st.session_state.chat_history:
    st.info("Ask a question below, or tap an example prompt in the sidebar to get started.")

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about alloy selection, properties, or applications…")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()

# Generate assistant reply if last message is from user
if (st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "user"):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        dataset_context=build_dataset_context(df)
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.chat_history)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            stream = client.chat.stream(model=model, messages=messages)
            for chunk in stream:
                try:
                    delta = chunk.data.choices[0].delta.content
                except AttributeError:
                    delta = None
                if delta:
                    full += delta
                    placeholder.markdown(full + " ▌")
            placeholder.markdown(full)
        except Exception as e:
            full = f"LLM error: `{type(e).__name__}` — {e}"
            placeholder.error(full)

    st.session_state.chat_history.append({"role": "assistant", "content": full})
    st.rerun()

footer()
