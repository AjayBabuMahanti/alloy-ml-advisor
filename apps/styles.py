"""
AlloyML shared UI module.

Design: clean, professional, SaaS-style.
- Single indigo accent on a light slate background.
- Subtle elevation (shadows + borders), hover states on interactive cards.
- Branded top strip, accent-bar section headers, and a consistent footer.

Every page should:
  1. call st.set_page_config(...)
  2. call apply_theme()
  3. call brand_bar()          # top strip with logo + nav label
  4. render content
  5. call footer()             # bottom credits
"""
import streamlit as st

# ---- design tokens --------------------------------------------------------
PRIMARY      = "#4f46e5"  # indigo-600
PRIMARY_SOFT = "#6366f1"  # indigo-500
PRIMARY_DARK = "#3730a3"  # indigo-800
PRIMARY_BG   = "#eef2ff"  # indigo-50
TEXT         = "#0f172a"  # slate-900
MUTED        = "#475569"  # slate-600
MUTED_SOFT   = "#94a3b8"  # slate-400
BORDER       = "#e2e8f0"  # slate-200
BORDER_SOFT  = "#f1f5f9"  # slate-100
SURFACE      = "#ffffff"
BG           = "#f8fafc"  # slate-50
SUCCESS      = "#059669"  # emerald-600
SUCCESS_BG   = "#d1fae5"  # emerald-100
WARN         = "#d97706"  # amber-600
WARN_BG      = "#fef3c7"  # amber-100
ERROR        = "#dc2626"  # red-600
ERROR_BG     = "#fee2e2"  # red-100

CHART_SEQ = [PRIMARY, "#14b8a6", "#f59e0b", "#ec4899", "#8b5cf6", "#0ea5e9"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
              size=13, color=TEXT),
    colorway=CHART_SEQ,
    margin=dict(l=40, r=20, t=50, b=40),
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    xaxis=dict(gridcolor=BORDER_SOFT, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER_SOFT, zerolinecolor=BORDER),
)


# ---- global theme ---------------------------------------------------------
def apply_theme():
    """Inject the unified stylesheet. Call once per page after set_page_config."""
    st.markdown(f"""
    <style>
    /* ---- base ---------------------------------------------------------- */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
        color: {TEXT};
    }}
    [data-testid="stAppViewContainer"] > .main {{ background: {BG}; }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        max-width: 1240px;
    }}
    /* hide default streamlit header but keep hamburger for nav */
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
    }}

    /* ---- headings ------------------------------------------------------ */
    h1 {{ color: {TEXT}; font-weight: 700; font-size: 1.85rem; letter-spacing: -0.02em;
          margin: 0 0 0.3rem 0; }}
    h2 {{ color: {TEXT}; font-weight: 600; font-size: 1.25rem; margin: 1.8rem 0 0.8rem 0; }}
    h3 {{ color: {TEXT}; font-weight: 600; font-size: 1.05rem; margin-top: 1.2rem; }}

    /* ---- sidebar ------------------------------------------------------- */
    [data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 0.5rem; }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {TEXT} !important; font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] h2 {{ font-size: 0.8rem !important; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED} !important; border: none; margin-top: 1.4rem; }}
    [data-testid="stSidebar"] label {{
        font-weight: 500 !important; font-size: 0.85rem !important; color: {TEXT} !important;
    }}

    /* ---- inputs -------------------------------------------------------- */
    .stNumberInput input, .stTextInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        background: {SURFACE} !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px {PRIMARY_BG} !important;
        outline: none !important;
    }}

    /* ---- buttons ------------------------------------------------------- */
    .stButton > button, .stDownloadButton > button {{
        background: {PRIMARY} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.3rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(79, 70, 229, 0.18) !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {PRIMARY_SOFT} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28) !important;
    }}
    .stButton > button:active {{ transform: translateY(0); }}
    .stButton > button:disabled {{
        background: {BORDER} !important;
        color: {MUTED_SOFT} !important;
        box-shadow: none !important;
    }}
    /* secondary buttons — sidebar example prompts */
    [data-testid="stSidebar"] .stButton > button {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        font-weight: 400 !important;
        text-align: left !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {PRIMARY_BG} !important;
        border-color: {PRIMARY} !important;
        color: {PRIMARY_DARK} !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* ---- metrics ------------------------------------------------------- */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {PRIMARY};
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {MUTED} !important; font-size: 0.72rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.08em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important; font-weight: 700 !important; font-size: 1.55rem !important;
    }}
    [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

    /* ---- data + tables ------------------------------------------------- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ---- expanders, alerts, chat -------------------------------------- */
    .streamlit-expanderHeader {{ font-weight: 500; color: {TEXT}; }}
    [data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid {BORDER}; }}
    [data-testid="stChatMessage"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }}
    [data-testid="stChatInput"] textarea {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}

    /* ---- misc ---------------------------------------------------------- */
    [data-testid="stHeaderActionElements"] {{ display: none !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{ color: {MUTED} !important; }}

    /* ---- custom classes ------------------------------------------------ */
    .ml-brand-bar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.6rem 0; margin-bottom: 1.2rem;
        border-bottom: 1px solid {BORDER};
    }}
    .ml-brand {{ display: flex; align-items: center; gap: 0.6rem; }}
    .ml-logo {{
        width: 32px; height: 32px; border-radius: 8px;
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        display: inline-flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.95rem; letter-spacing: -0.02em;
        box-shadow: 0 2px 6px rgba(79,70,229,0.25);
    }}
    .ml-brand-name {{ font-size: 1.02rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.01em; }}
    .ml-brand-tag  {{ font-size: 0.78rem; color: {MUTED}; font-weight: 500;
                      padding: 0.15rem 0.55rem; background: {BORDER_SOFT};
                      border-radius: 6px; margin-left: 0.35rem; }}
    .ml-crumb {{ font-size: 0.82rem; color: {MUTED}; font-weight: 500; }}
    .ml-crumb-cur {{ color: {TEXT}; }}

    .ml-hero-title {{ font-size: 2.2rem; font-weight: 700; color: {TEXT};
                      letter-spacing: -0.025em; margin: 0 0 0.5rem 0; line-height: 1.15; }}
    .ml-hero-sub   {{ font-size: 1.02rem; color: {MUTED}; max-width: 640px;
                      margin: 0 0 1.5rem 0; line-height: 1.55; }}
    .ml-stat-band  {{
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;
        margin: 1rem 0 2rem 0;
    }}
    .ml-stat {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }}
    .ml-stat-label {{ font-size: 0.72rem; color: {MUTED}; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }}
    .ml-stat-value {{ font-size: 1.35rem; font-weight: 700; color: {TEXT}; }}
    .ml-stat-hint  {{ font-size: 0.72rem; color: {MUTED_SOFT}; margin-top: 0.15rem; }}

    .ml-section-title {{
        font-size: 1.2rem; font-weight: 600; color: {TEXT};
        padding-left: 0.75rem; margin: 1.8rem 0 0.8rem 0;
        border-left: 3px solid {PRIMARY};
        line-height: 1.3;
    }}

    .ml-nav-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 1.4rem; height: 100%; transition: all 0.18s ease;
        position: relative; overflow: hidden;
    }}
    .ml-nav-card:hover {{
        border-color: {PRIMARY};
        box-shadow: 0 6px 18px rgba(15,23,42,0.08);
        transform: translateY(-2px);
    }}
    .ml-nav-icon {{
        width: 40px; height: 40px; border-radius: 10px;
        background: {PRIMARY_BG};
        display: inline-flex; align-items: center; justify-content: center;
        color: {PRIMARY}; font-size: 1.3rem; font-weight: 700;
        margin-bottom: 0.9rem;
    }}
    .ml-nav-title {{ font-size: 1.05rem; font-weight: 600; color: {TEXT};
                     margin-bottom: 0.35rem; }}
    .ml-nav-body {{ font-size: 0.88rem; color: {MUTED}; line-height: 1.5; }}

    .ml-badge {{
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.01em;
    }}
    .ml-badge-primary {{ background: {PRIMARY_BG}; color: {PRIMARY_DARK}; }}
    .ml-badge-success {{ background: {SUCCESS_BG}; color: {SUCCESS}; }}
    .ml-badge-warn    {{ background: {WARN_BG};    color: {WARN};    }}
    .ml-badge-muted   {{ background: {BORDER_SOFT}; color: {MUTED}; }}

    .ml-footer {{
        margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid {BORDER};
        color: {MUTED_SOFT}; font-size: 0.82rem; line-height: 1.5;
        display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---- layout helpers -------------------------------------------------------
def brand_bar(page_label: str = ""):
    """Top branded strip with logo + current page breadcrumb."""
    crumb = ""
    if page_label:
        crumb = (f'<div class="ml-crumb">AlloyML  ›  '
                 f'<span class="ml-crumb-cur">{page_label}</span></div>')
    st.markdown(f"""
    <div class="ml-brand-bar">
      <div class="ml-brand">
        <span class="ml-logo">Al</span>
        <span class="ml-brand-name">AlloyML</span>
        <span class="ml-brand-tag">IIT Roorkee · BTP 2025-26</span>
      </div>
      {crumb}
    </div>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Page title + subtitle block."""
    sub = f'<p class="ml-hero-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:0.2rem 0 1.2rem 0;">
      <h1 class="ml-hero-title">{title}</h1>
      {sub}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    """Section header with a left accent bar."""
    st.markdown(f'<div class="ml-section-title">{title}</div>',
                unsafe_allow_html=True)


def stat_band(stats: list):
    """Compact grid of labeled stats. Each item: (label, value, hint_opt)."""
    cells = []
    for item in stats:
        label, value = item[0], item[1]
        hint = item[2] if len(item) > 2 else ""
        hint_html = f'<div class="ml-stat-hint">{hint}</div>' if hint else ""
        cells.append(
            f'<div class="ml-stat">'
            f'<div class="ml-stat-label">{label}</div>'
            f'<div class="ml-stat-value">{value}</div>'
            f'{hint_html}'
            f'</div>'
        )
    st.markdown('<div class="ml-stat-band">' + "".join(cells) + '</div>',
                unsafe_allow_html=True)


def nav_card(title: str, body: str, icon: str = ""):
    """Navigation card HTML. Render inside a st.columns() cell."""
    return f"""
    <div class="ml-nav-card">
      <div class="ml-nav-icon">{icon}</div>
      <div class="ml-nav-title">{title}</div>
      <div class="ml-nav-body">{body}</div>
    </div>
    """


def badge(text: str, kind: str = "primary"):
    """Inline pill badge. kind: primary | success | warn | muted."""
    return f'<span class="ml-badge ml-badge-{kind}">{text}</span>'


def footer():
    """Consistent footer across pages."""
    st.markdown(f"""
    <div class="ml-footer">
      <div>
        <strong style="color:{MUTED};">AlloyML</strong> · Final Year BTP · MIN-400A
        · IIT Roorkee, Dept. of Mechanical &amp; Industrial Engineering
      </div>
      <div>Models: LightGBM · RandomForest · Mistral LLM</div>
    </div>
    """, unsafe_allow_html=True)
