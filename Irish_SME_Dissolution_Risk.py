"""
Irish SME Dissolution Risk Dashboard

Early-warning triage for Irish SME dissolution, scored from public CRO
filing-compliance metadata alone. Stage 1 is a cost-sensitive XGBoost
classifier; Stage 2 an Isolation Forest anomaly detector; the two combine into
four operational risk tiers. A language-model layer explains each score in
plain English and classifies entity type.

    streamlit run Irish_SME_Dissolution_Risk.py

Reads its data from data/processed/ and outputs/. Every section degrades to a
message rather than a traceback when a file is absent, so the app runs against
a partial pipeline.

Optional: OPENAI_API_KEY in .env (locally) or in Streamlit secrets (deployed)
for live narrative generation. Without one, saved narratives are still shown.
"""

import os
import re
import pathlib
from datetime import datetime
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go


# Report and export generators are imported on first use, not at startup, so
# reportlab and matplotlib stay out of the launch path.
_THIS_FILE = pathlib.Path(__file__).name
_REPORT_OK = (pathlib.Path(__file__).resolve().parent / "per_company_report.py").exists()
_REPORT_ERR = ("" if _REPORT_OK else
               f"per_company_report.py not found next to {_THIS_FILE}")


def _report_api():
    """Import the per-company report generator on first use."""
    from per_company_report import (build_company_pdf,
                                    load_sources as _load_report_sources,
                                    report_filename)
    return build_company_pdf, _load_report_sources, report_filename

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
ROOT = pathlib.Path(__file__).resolve().parent
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

UI_YELLOW = "#FFE600"
UI_DARK = "#2E2E38"
UI_MID = "#3D3D4E"
UI_BG = "#1A1A23"
UI_TEXT_DIM = "#A0A0B0"

TIER_ORDER = ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]

TIER_DESCRIPTIONS = {
    "PRIORITY":
        "Stage 1 High (top 5% by calibrated dissolution risk) AND flagged as a "
        "behavioural anomaly by Isolation Forest. Both signals agree. This is an "
        "intersection, not a cut-off: 78 of the 1,532 High companies were also "
        "anomaly-flagged.",
    "DISSOLUTION_RISK":
        "Stage 1 High or Medium (top 20% by calibrated dissolution risk), "
        "without an anomaly flag. Not a lower tier: this holds 1,454 companies "
        "in the same top-5% score band as PRIORITY, separated only by Stage 2.",
    "BEHAVIORAL_ANOMALY":
        "Flagged as behavioural anomaly by Isolation Forest but not in the supervised "
        "high-risk band. Filing pattern unusual, worth a watch.",
    "LOW_CONCERN":
        "Bottom 80% of supervised risk and not flagged by Isolation Forest. "
        "No signal from either stage.",
}


TIER_COLORS = {
    "PRIORITY": "#C1121F",
    "DISSOLUTION_RISK": "#E07B00",
    "BEHAVIORAL_ANOMALY": "#1A73E8",
    "LOW_CONCERN": "#1A7340",
}

st.set_page_config(
    page_title="Dissolution Risk Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Entrance motion runs on first load only. Streamlit re-executes this script on
# every widget interaction, so an ungated CSS animation re-fires each time and the
# whole page flickers on every click. The flag is set once per session.
_FIRST_PAINT = "ui_painted" not in st.session_state
st.session_state["ui_painted"] = True
_ENTER = "enter" if _FIRST_PAINT else ""

def _css_min(css: str) -> str:
    """Collapse a stylesheet onto one line before it reaches st.markdown.

    Streamlit's markdown parser follows CommonMark, where a raw HTML block ends
    at the first blank line. A <style> block written with blank lines between
    sections is therefore truncated at the first one, and everything after it is
    printed onto the page as text. Whitespace is not significant in CSS, so
    collapsing it is safe and removes the failure mode entirely.
    """
    return " ".join(css.split())


_CSS = f"""
:root {{
  --mono: "IBM Plex Mono","SFMono-Regular",Consolas,monospace;
  --sans: "IBM Plex Sans","Segoe UI",sans-serif;
  --ease: cubic-bezier(.2,.7,.3,1);
}}
/* A warm bloom off the top-left and a cool one top-right, over near-black, with
   a faint measured grid masked to fade at the edges. Depth, not decoration: the
   glass panels above it need something to sit on or they read as flat boxes. */
.stApp {{
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(255,230,0,.09), transparent 62%),
    radial-gradient(900px 480px at 96% 2%, rgba(61,61,78,.5), transparent 60%),
    linear-gradient(180deg, {UI_BG} 0%, #15151C 100%);
  background-attachment: fixed;
}}
.stApp::before {{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(circle at 50% 30%, #000 12%, transparent 78%);
  -webkit-mask-image: radial-gradient(circle at 50% 30%, #000 12%, transparent 78%);
}}
.block-container {{ padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px;
                    position: relative; z-index: 1; }}
h1, h2, h3, h4, h5 {{ color: #FFFFFF; font-family: var(--sans); }}
h1 {{ letter-spacing: -0.5px; font-weight: 700; }}
h5 {{ font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .1em;
      color: {UI_TEXT_DIM} !important; font-weight: 600 !important; }}
/* Every figure is a measurement, so every figure is monospaced and column
   aligned. tabular-nums stops digits shifting sideways as values change. */
[data-testid="stMetricValue"], .rq-card .v, .stDataFrame td, code,
.profile-row span:last-child {{
  font-family: var(--mono) !important;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}}
@keyframes rise {{ from {{ opacity:0; transform:translateY(9px); }}
                    to {{ opacity:1; transform:none; }} }}
@keyframes breathe {{ 0%,100% {{ opacity:.35; transform:scale(1); }}
                       50% {{ opacity:1; transform:scale(1.35); }} }}
.enter > div {{ animation: rise .45s var(--ease) both; }}
.enter > div:nth-child(2) {{ animation-delay:.05s; }}
.enter > div:nth-child(3) {{ animation-delay:.10s; }}
.enter > div:nth-child(4) {{ animation-delay:.15s; }}
@media (prefers-reduced-motion: reduce) {{
  .enter > div, [data-testid="stMetric"] {{ animation: none !important; }}
}}
/* Glass: a translucent plate with a hairline top edge where light would catch. */
[data-testid="stMetric"], .rq-card, .shap-driver, .narrative-box,
div[data-testid="stExpander"] {{
  background: linear-gradient(155deg, rgba(255,255,255,.055), rgba(255,255,255,.012));
  backdrop-filter: blur(12px) saturate(112%);
  -webkit-backdrop-filter: blur(12px) saturate(112%);
  border-top: 1px solid rgba(255,255,255,.09);
  border-right: 1px solid rgba(255,255,255,.04);
  border-bottom: 1px solid rgba(255,255,255,.04);
  box-shadow: 0 8px 28px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.09);
}}
[data-testid="stMetric"] {{
  padding: 14px 18px; border-radius: 10px;
  border-left: 3px solid {UI_YELLOW};
  transition: transform .18s var(--ease), border-left-width .18s var(--ease),
              box-shadow .18s var(--ease), border-color .18s var(--ease);
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-2px); border-left-width: 6px;
  border-top-color: rgba(255,230,0,.28);
  box-shadow: 0 14px 38px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.15),
              0 0 0 1px rgba(255,230,0,.12);
}}
[data-testid="stMetric"] label {{
  color: {UI_TEXT_DIM} !important; font-size: 0.64rem !important;
  text-transform: uppercase; letter-spacing: .105em; font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{ color: #FFFFFF !important; font-size: 1.62rem !important; }}
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; padding: 6px; border-radius: 10px;
  background: linear-gradient(155deg, rgba(255,255,255,.05), rgba(255,255,255,.01));
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.07);
}}
.stTabs [data-baseweb="tab"] {{
  color: {UI_TEXT_DIM}; font-weight: 500; padding: 10px 22px; border-radius: 7px;
  transition: background .16s var(--ease), color .16s var(--ease);
}}
.stTabs [data-baseweb="tab"]:hover {{ background: rgba(255,255,255,.06); color: #FFF; }}
.stTabs [aria-selected="true"] {{
  background: {UI_YELLOW} !important; color: {UI_DARK} !important;
  box-shadow: 0 2px 14px rgba(255,230,0,.22);
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
div[data-testid="stExpander"] {{ border-radius: 10px; }}
div[data-testid="stExpander"] summary {{ color: #FFF; }}
.stDataFrame {{ border: 1px solid rgba(255,255,255,.08); border-radius: 8px; }}
.stDataFrame tbody tr:hover td {{ background: rgba(255,255,255,.05) !important; }}
.profile-row {{
  display:flex;justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,.07);padding:7px 0;
  transition: background .14s var(--ease), padding-left .14s var(--ease);
}}
.profile-row:hover {{ background: rgba(255,255,255,.035); padding-left: 6px; }}
.profile-row span:first-child {{ color: {UI_TEXT_DIM}; font-size:0.85rem;
                                 font-family: var(--sans); }}
.profile-row span:last-child  {{ color: #FFFFFF; font-weight: 500; }}
.rq-card {{
  padding:18px;border-radius:10px;border-left:3px solid #1A7340;height:130px;
  transition: transform .18s var(--ease), box-shadow .18s var(--ease),
              border-left-width .18s var(--ease);
}}
.rq-card:hover {{
  transform: translateY(-2px); border-left-width: 6px;
  box-shadow: 0 14px 38px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.15);
}}
.rq-card .h {{ color:#1A7340;font-weight:600;font-size:0.72rem;
               text-transform:uppercase;letter-spacing:.09em; }}
.rq-card .v {{ font-size:1.6rem;color:#FFF;margin:6px 0; }}
.rq-card .s {{ color:{UI_TEXT_DIM};font-size:0.76rem; }}
.shap-driver {{
  padding:12px 16px;border-radius:8px;border-left:3px solid {UI_YELLOW};
  margin-bottom:10px;
  transition: border-left-width .18s var(--ease), transform .18s var(--ease),
              box-shadow .18s var(--ease);
}}
.shap-driver:hover {{
  border-left-width:7px; transform: translateX(3px);
  box-shadow: 0 10px 30px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.14);
}}
.narrative-box {{
  padding:18px 20px;border-left:3px solid {UI_YELLOW};
  border-radius:8px;color:#FFF;line-height:1.65;font-family: var(--sans);
}}
.pulse {{
  display:inline-block;width:6px;height:6px;border-radius:50%;
  margin-right:7px;vertical-align:middle;
  animation: breathe 2.6s ease-in-out infinite;
}}
"""

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
    "<style>" + _css_min(_CSS) + "</style>",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================
def to_float(v) -> float:
    """Coerce any value to float. Handles bracket-wrapped array repr like '[0.499]'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return to_float(v[0]) if len(v) else np.nan
    s = str(v).strip()
    if s in ("", "nan", "None", "[]"):
        return np.nan
    s = s.strip("[]").strip()
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.split()[0])
        except Exception:
            return np.nan


def safe_int(v) -> int:
    f = to_float(v)
    return 0 if np.isnan(f) else int(f)


def fmt_pct(v) -> str:
    f = to_float(v)
    return "n/a" if np.isnan(f) else f"{f*100:.1f}%"


def plotly_dark(fig, height: int = 400, margin=None, **kwargs):
    """Apply the themed dark layout. `margin` is an explicit kwarg so callers
    can override the default without a duplicate-keyword TypeError."""
    if margin is None:
        margin = dict(t=20, b=40, l=40, r=20)
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=margin,
        font=dict(family="IBM Plex Sans, Segoe UI", color="#FFF"),
        # Figures are measurements, so numerals are monospaced and column-aligned
        # wherever Plotly draws them: axis ticks, bar labels, hover.
        hoverlabel=dict(bgcolor=UI_DARK, bordercolor=UI_YELLOW,
                        font=dict(family="IBM Plex Mono, monospace", color="#FFF",
                                  size=12)),
        # A value that changes should be seen to change rather than jump.
        transition=dict(duration=320, easing="cubic-in-out"),
    )
    # Caller-supplied kwargs (showlegend, xaxis_title, etc.) win
    layout.update(kwargs)
    fig.update_layout(**layout)
    fig.update_xaxes(tickfont=dict(family="IBM Plex Mono, monospace", size=11),
                     gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.14)")
    fig.update_yaxes(tickfont=dict(family="IBM Plex Mono, monospace", size=11),
                     gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.14)")
    return fig


def safe_block(label: str = "section"):
    """Context manager: catch any exception in a tab section and show it inline
    instead of crashing the whole dashboard."""
    class _SafeBlock:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                st.warning(f"{label}: {exc_type.__name__}: {exc_val}")
                return True  # suppress
            return False
    return _SafeBlock()


# Values treated as "no real category" - excluded from sector/county breakdowns
UNKNOWN_TOKENS = {"", "unknown", "UNKNOWN", "Unknown", "U", "u",
                  "nan", "NaN", "None", "\u2014", "-", "N/A", "n/a"}


# Plain-language labels for model features. Displayed as "Plain label (raw_name)"
# so an auditor reads the business term while the underlying field stays traceable.
FEATURE_LABELS = {
    "ar_filed_count": "Annual returns filed",
    "annual_submission_rate": "Annual submission rate",
    "total_submissions": "Total submissions",
    "director_change_count": "Director changes",
    "company_age_years": "Company age (years)",
    "submission_history_years": "Filing history length",
    "other_form_count": "Other filings",
    "age_vs_sector_median": "Age vs sector median",
    "days_since_last_name_change": "Time since last name change",
    "name_change_count": "Name changes",
    "days_since_last_office_change": "Time since last office change",
    "days_since_last_charge": "Time since last charge",
    "outstanding_charge_count": "Outstanding charges",
    "charge_count": "Charges registered",
    "office_change_count": "Registered office changes",
    "name_risk_score": "Name risk score",
    "sector_relative_deviation": "Sector-relative deviation",
    "filing_consistency": "Filing consistency",
    "avg_days_to_file": "Average days to file",
    "director_dissolution_count": "Director prior dissolutions",
    "director_avg_portfolio_size": "Director portfolio size",
}


def feature_label(name: str) -> str:
    """Return 'Plain label (raw_name)' for a feature, or the raw name if unmapped."""
    plain = FEATURE_LABELS.get(str(name))
    return f"{plain} ({name})" if plain else str(name)


def _signal_bullets(signals: str) -> str:
    """Turn a semicolon-separated distress-signal string into a bullet block."""
    items = [s.strip() for s in str(signals).split(";") if s.strip()]
    if not items:
        return ""
    return "\n\nKey risk indicators:\n" + "\n".join(f"- {s}" for s in items)


def band_txt(v) -> str:
    """A position within the cohort, rendered as a band. Deliberately distinct
    from pct_txt: this is a rank among companies, not a probability of anything,
    and the two must never be read as the same kind of number."""
    f = to_float(v)
    return "n/a" if np.isnan(f) else f"Top {f * 100:.1f}%"


def pct_txt(v) -> str:
    """Format a calibrated probability. Values that are small but not zero must
    not render as '0.0%': at four decimal places 0.0004 is a real probability and
    displaying it as zero reads as certainty the company will not dissolve."""
    f = to_float(v)
    if np.isnan(f):
        return "n/a"
    if f <= 0.0:
        return "0.0%"
    if f < 0.001:
        return "<0.1%"
    return f"{f * 100:.1f}%"


def company_key(v) -> str:
    """Normalize a company number so int/float/zero-padded forms all match."""
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip().lstrip("0") or "0"


def drop_unknowns(series: pd.Series) -> pd.Series:
    """Drop NaN/blank/'Unknown' entries from a categorical series so breakdowns
    show only meaningful counties/sectors."""
    s = series.dropna()
    s = s.astype(str).str.strip()
    return s[~s.isin(UNKNOWN_TOKENS) & (s.str.len() > 0)]


_NACE_SECTION_RANGES = [
    (1, 3, "Agriculture, Forestry & Fishing"), (5, 9, "Mining & Quarrying"),
    (10, 33, "Manufacturing"), (35, 35, "Electricity & Gas"),
    (36, 39, "Water & Waste Management"), (41, 43, "Construction"),
    (45, 47, "Wholesale & Retail Trade"), (49, 53, "Transportation & Storage"),
    (55, 56, "Accommodation & Food Service"), (58, 63, "Information & Communication"),
    (64, 66, "Financial & Insurance"), (68, 68, "Real Estate"),
    (69, 75, "Professional, Scientific & Technical"),
    (77, 82, "Administrative & Support Service"),
    (84, 84, "Public Administration & Defence"), (85, 85, "Education"),
    (86, 88, "Human Health & Social Work"),
    (90, 93, "Arts, Entertainment & Recreation"),
    (94, 96, "Other Service Activities"), (97, 98, "Household Employers"),
    (99, 99, "Extraterritorial Organisations"),
]


def nace_digits(code) -> str:
    """Normalise a NACE Rev.2 class code to four digits.

    The column is float-typed on read, so a code beginning with zero loses it:
    0141 becomes 141.0. Slicing the first two characters of that gives 14 and
    maps a dairy farm to Manufacturing. Any code shorter than four digits is
    therefore a code whose leading zeros were destroyed, and is restored.
    Returns '' for UNKNOWN and other non-numeric values.
    """
    s = re.sub(r"[^0-9]", "", str(code).strip().split(".")[0])
    if not s or int(s) == 0:
        return ""
    return s.zfill(4) if len(s) < 4 else s[:4]


def nace_section_label(code) -> str:
    """Map a NACE Rev.2 code to a readable section label; '' if unknown."""
    s = nace_digits(code)
    if len(s) < 2:
        return ""
    d = int(s[:2])
    for lo, hi, label in _NACE_SECTION_RANGES:
        if lo <= d <= hi:
            return label
    return ""


def add_sector_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a correctly derived 'sector_label' from raw nace_v2_code."""
    if "nace_v2_code" in df.columns:
        df = df.copy()
        df["sector_label"] = df["nace_v2_code"].apply(nace_section_label)
    return df


# =============================================================================
# CACHED LOADERS
# =============================================================================
def _scrub_bracket_columns(df: pd.DataFrame) -> pd.DataFrame:
    """For any non-numeric column whose values look like '[0.499]', coerce the
    entire column to numeric via to_float. This neutralises a CSV-roundtrip bug
    where a 1-element numpy array was saved with its bracketed repr.
    Cheap: only checks the first 20 non-null values per column."""
    from pandas.api.types import is_numeric_dtype
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            continue
        sample = df[c].dropna().head(20).astype(str)
        if sample.empty:
            continue
        has_brackets = sample.str.startswith("[") & sample.str.endswith("]")
        if has_brackets.any():
            df[c] = df[c].apply(to_float)
    return df


@st.cache_data(show_spinner=False)
def load_prospective() -> pd.DataFrame:
    path = DATA_PROC / "prospective_final.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = _scrub_bracket_columns(df)
    # Always coerce known score columns regardless of dtype
    for c in ("dissolution_risk_score", "if_anomaly_score",
              "lof_anomaly_score"):
        if c in df.columns:
            df[c] = df[c].apply(to_float)
    return df


@st.cache_data(show_spinner=False)
def load_priority() -> pd.DataFrame:
    path = TABLES / "step5_priority_companies.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = _scrub_bracket_columns(df)
    for c in ("dissolution_risk_score", "if_anomaly_score"):
        if c in df.columns:
            df[c] = df[c].apply(to_float)
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path = TABLES / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_winner_meta() -> dict:
    path = MODELS / "winner_meta.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(k): v for k, v in zip(df["key"], df["value"])}


@st.cache_data(show_spinner=False)
def load_feature_cols() -> List[str]:
    path = MODELS / "feature_cols.txt"
    if not path.exists():
        return []
    return [l.strip() for l in open(path) if l.strip()]


@st.cache_data(show_spinner=False)
def load_narratives_dict() -> dict:
    """Load per-company summaries from the Stage 2 LLM output (llm_features.csv),
    keyed by a normalized company number so int/float/string forms all match.
    Falls back to the legacy narrative text file if the CSV is absent."""
    def num_key(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip().lstrip("0") or "0"

    out = {}
    csv_path = OUTPUTS / "nlp" / "llm_features.csv"
    if csv_path.exists():
        try:
            wanted = ("company_num", "audit_narrative", "distress_signals")
            df = pd.read_csv(csv_path, low_memory=False,
                             usecols=lambda c: c in wanted)
            if {"company_num", "audit_narrative"}.issubset(df.columns):
                narr = df["audit_narrative"].fillna("").astype(str).str.strip()
                keep = (df["company_num"].notna() & narr.ne("")
                        & narr.str.lower().ne("nan"))
                blocks = narr[keep]
                if "distress_signals" in df.columns:
                    sig = df.loc[keep, "distress_signals"].fillna("").astype(str).str.strip()
                    sig = sig.where(sig.str.lower().ne("nan"), "")
                    blocks = blocks + sig.map(_signal_bullets)
                out = dict(zip(df.loc[keep, "company_num"].map(num_key), blocks))
            if out:
                return out
        except Exception:
            pass

    # Legacy fallback.
    path = TABLES / "step6_shap_narratives.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("=" * 72)
    for block in blocks:
        m = re.search(r"Company Number\s*:\s*(\S+)", block)
        if m:
            out[num_key(m.group(1).strip())] = block.strip()
    return out


@st.cache_data(show_spinner=False)
def load_shap_global() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = OUTPUTS / "step6_shap_values_test.npy"
    if not path.exists():
        return None, None
    vals = np.load(path)
    return vals, np.abs(vals).mean(axis=0)


ENTITY_LABELS = {
    "special_purpose_vehicle": "Special purpose vehicle",
    "holding_company": "Holding company",
    "trading_business": "Trading business",
    "uncertain": "Unclassified",
}


@st.cache_data(show_spinner=False)
def load_entity_types() -> Optional[dict]:
    """Load the entity classification per company from the nlp_06 output.
    This is the same classification reported in the dissertation, so the
    dashboard and the write-up agree on how many companies are SPVs.
    Returns {normalized_company_num: label} or None if absent."""
    path = OUTPUTS / "nlp" / "entity_types.csv"
    if not path.exists():
        return None
    try:
        head = pd.read_csv(path, nrows=1)
        col = next((c for c in ("entity_type", "entity", "type", "classification",
                                "label", "predicted_type")
                    if c in head.columns), None)
        if col is None or "company_num" not in head.columns:
            return None
        df = pd.read_csv(path, low_memory=False, usecols=["company_num", col])
        raw = df[col].fillna("uncertain").astype(str).str.strip().str.lower()
        labelled = raw.map(ENTITY_LABELS).fillna("Unclassified")
        return dict(zip(df["company_num"].map(company_key), labelled))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_report_sources():
    """Data frames the per-company PDF report needs. None if unavailable."""
    if not _REPORT_OK:
        return None
    try:
        _, _load_report_sources, _ = _report_api()
        return _load_report_sources(
            shap=OUTPUTS / "prospective_shap.csv",
            llm=OUTPUTS / "nlp" / "llm_features.csv",
            spv=OUTPUTS / "nlp" / "prospective_spv_labelled.csv",
            entity=OUTPUTS / "nlp" / "entity_types.csv",
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_model_vs_llm() -> Optional[dict]:
    """nlp_09 output: an independent LLM's tier call per company, model tier withheld.
    Returns {company_key: llm_tier_string} or None. Loaded lazily, not at startup."""
    path = OUTPUTS / "nlp" / "model_vs_llm.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if "company_num" not in df.columns:
            return None
        tier_col = next((c for c in ("llm_tier", "llm_risk_tier", "llm_label",
                                     "llm_pred_tier", "llm_assessment",
                                     "llm_combined_risk_tier", "predicted_tier")
                         if c in df.columns), None)
        if tier_col is None:
            return None
        keys = df["company_num"].map(company_key)
        return dict(zip(keys, df[tier_col].astype(str)))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_concordance_summary() -> Optional[dict]:
    """nlp_08: mean faithfulness of AI narratives to SHAP drivers. None if absent."""
    path = OUTPUTS / "nlp" / "shap_llm_concordance.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        out = {}
        for col in ("precision_top5", "cites_any_top3", "leads_with_top_driver"):
            if col in df.columns:
                out[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        return out or None
    except Exception:
        return None


def _patch_xgb_base_score(model):
    """Patch base_score everywhere XGBoost stashes it:
    1. The XGBClassifier wrapper's .base_score attribute (sklearn-side)
    2. The wrapper's .intercept_ if present
    3. The booster's internal JSON config
    Returns (model, scalar_value_or_None)."""
    scalar = None
    # 1. Wrapper attribute
    try:
        if hasattr(model, "base_score"):
            current = model.base_score
            if isinstance(current, str) and current.strip().startswith("["):
                inner = current.strip().strip("[]").strip().split(",")[0].strip()
                scalar = float(inner)
                model.base_score = scalar
            elif isinstance(current, (list, tuple)) and len(current) > 0:
                scalar = float(current[0])
                model.base_score = scalar
            elif hasattr(current, "__len__") and len(current) > 0:
                scalar = float(current[0])
                model.base_score = scalar
    except Exception:
        pass
    # 2. intercept_
    try:
        if hasattr(model, "intercept_"):
            ic = model.intercept_
            if isinstance(ic, str) and ic.strip().startswith("["):
                inner = ic.strip().strip("[]").strip().split(",")[0].strip()
                model.intercept_ = float(inner)
                if scalar is None:
                    scalar = float(inner)
    except Exception:
        pass
    # 3. Booster config
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        import json as _json
        cfg = _json.loads(booster.save_config())
        learner = cfg.get("learner", {})
        mparam = learner.get("learner_model_param", {})
        bs = mparam.get("base_score", None)
        if isinstance(bs, str) and bs.strip().startswith("["):
            inner = bs.strip().strip("[]").strip().split(",")[0].strip()
            sval = float(inner)
            mparam["base_score"] = f"{sval}"
            booster.load_config(_json.dumps(cfg))
            if scalar is None:
                scalar = sval
    except Exception:
        pass
    return model, scalar


class NativeXGBExplainer:
    """Drop-in TreeExplainer replacement that uses XGBoost's built-in TreeSHAP
    (booster.predict(pred_contribs=True)). Bypasses the shap library entirely,
    so it doesn't hit the base_score parsing bug."""

    def __init__(self, model):
        import xgboost as _xgb
        self._xgb = _xgb
        self.booster = (model.get_booster()
                        if hasattr(model, "get_booster") else model)
        self.expected_value = 0.0  # populated lazily after first call

    def shap_values(self, X, check_additivity=False):
        if hasattr(X, "values"):
            X = X.values
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = self._xgb.DMatrix(X.astype(np.float32))
        contribs = self.booster.predict(dm, pred_contribs=True)
        # Last column is the base value (constant across all rows)
        if contribs.ndim == 2:
            self.expected_value = float(contribs[0, -1])
            return contribs[:, :-1]
        else:
            # Multi-class case - shouldn't happen for binary classifier
            return contribs[..., :-1]


@st.cache_resource(show_spinner=False)
def load_model_artefacts():
    """Load winner model + isotonic calibrator + SHAP TreeExplainer.
    Returns (model, calibrator, explainer, winner_name, explainer_error_message)."""
    meta = load_winner_meta()
    winner_name = str(meta.get("winner_name", "XGBoost"))
    fname = winner_name.lower().replace(" ", "_") + "_model.joblib"
    mpath = MODELS / fname
    if not mpath.exists():
        return None, None, None, winner_name, f"model file not found: {fname}"
    try:
        model = joblib.load(mpath)
    except Exception as e:
        return None, None, None, winner_name, f"joblib.load failed: {e}"
    calibrator = None
    if (MODELS / "isotonic_calibrator.joblib").exists():
        try:
            calibrator = joblib.load(MODELS / "isotonic_calibrator.joblib")
        except Exception:
            calibrator = None

    try:
        import shap as _shap
    except ImportError:
        _shap = None

    actual_model = model
    if hasattr(model, "steps"):
        actual_model = model.steps[-1][1]
    actual_model, patched = _patch_xgb_base_score(actual_model)

    errors = []
    explainer = None

    # Strategy 1: shap.TreeExplainer default
    if _shap is not None:
        try:
            explainer = _shap.TreeExplainer(actual_model)
            return model, calibrator, explainer, winner_name, None
        except Exception as e:
            errors.append(f"shap.TreeExplainer default: {type(e).__name__}: {e}")

    # Strategy 2: shap.TreeExplainer on booster
    if _shap is not None:
        try:
            booster = (actual_model.get_booster()
                       if hasattr(actual_model, "get_booster") else actual_model)
            explainer = _shap.TreeExplainer(booster)
            return model, calibrator, explainer, winner_name, None
        except Exception as e:
            errors.append(f"shap.TreeExplainer on booster: {type(e).__name__}: {e}")

    # Strategy 3: NATIVE XGBoost TreeSHAP (bypasses shap library and base_score)
    try:
        explainer = NativeXGBExplainer(actual_model)
        # Smoke test - run a single prediction to confirm it works
        feat_count = len(load_feature_cols()) or 84
        _ = explainer.shap_values(np.zeros((1, feat_count), dtype=np.float32))
        return model, calibrator, explainer, winner_name, None
    except Exception as e:
        errors.append(f"NativeXGBExplainer: {type(e).__name__}: {e}")

    # Strategy 4: shap.Explainer auto-dispatch
    if _shap is not None:
        try:
            explainer = _shap.Explainer(actual_model)
            return model, calibrator, explainer, winner_name, None
        except Exception as e:
            errors.append(f"shap.Explainer auto: {type(e).__name__}: {e}")

    suffix = f"\n(patched base_score scalar={patched})" if patched is not None else ""
    return model, calibrator, None, winner_name, "\n".join(errors) + suffix


def ollama_available(base_url: str = "http://localhost:11434", timeout: float = 1.5):
    """Detect if Ollama is running locally. Returns (list_of_models, status_msg)."""
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            return [], "running but no models pulled (run: ollama pull llama3.1:8b)"
        return models, f"running with {len(models)} model(s)"
    except Exception as e:
        return [], f"not running ({type(e).__name__})"


def ollama_generate(model: str, prompt: str,
                    base_url: str = "http://localhost:11434",
                    timeout: float = 120.0) -> str:
    """Generate narrative via local Ollama. Synchronous (stream=False)."""
    import urllib.request
    import json as _json
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 350},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def get_openai_client():
    """Lazy OpenAI client. Reads the API key from env or the .env file.
    Returns (client, status_message)."""
    candidates = ["OPENAI_API_KEY", "OPENAI_KEY"]
    key, src = "", ""
    for name in candidates:
        v = os.environ.get(name, "").strip()
        if v:
            key, src = v, f"env ({name})"
            break
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            try:
                for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().lstrip("export ").strip()
                    if k in candidates:
                        v = v.strip().strip('"').strip("'")
                        if v:
                            key, src = v, f".env ({k})"
                            break
            except Exception as e:
                return None, f".env read error: {e}"
    if not key:
        return None, "no API key in env or .env"
    try:
        import openai
    except ImportError:
        return None, "the openai package isn't installed (pip install openai)"
    try:
        return openai.OpenAI(api_key=key), f"loaded from {src}"
    except Exception as e:
        return None, f"client init failed: {e}"


def parse_drivers_from_narrative(narrative_text: str) -> List[Tuple[str, float, str]]:
    """Extract top SHAP drivers from a narrative block.
    Lines look like:
       1. ar_filed_count = 3              ->  ^ RISK   (|SHAP|=2.4109)
       2. company_age_years = 14.2        ->  v risk   (|SHAP|=1.8732)
    Returns list of (feature_name, signed_shap_value, feature_value_as_str)."""
    import re as _re
    pattern = _re.compile(
        r"\d+\.\s+(\S+)\s*=\s*(.+?)\s*->\s*([\^v])\s*[\w\s]+\(\|SHAP\|=([\d.]+)\)",
        flags=_re.IGNORECASE,
    )
    drivers = []
    for m in pattern.finditer(narrative_text):
        feat = m.group(1).strip()
        val_str = m.group(2).strip()
        direction = 1.0 if m.group(3) == "^" else -1.0
        shap_mag = float(m.group(4))
        drivers.append((feat, direction * shap_mag, val_str))
    return drivers


def get_drivers_for_company(row: pd.Series, narratives_dict: dict,
                            explainer, feature_cols: List[str],
                            mean_abs_shap: Optional[np.ndarray]
                            ) -> Tuple[Optional[List[Tuple[str, float, float]]], str]:
    """Get top-5 SHAP drivers for a company. Returns (drivers, source_description).
    Tries 3 sources in priority order:
      1. Pre-computed per-company SHAP from narratives file (most accurate)
      2. Live per-company SHAP via TreeExplainer (if working)
      3. Global mean-abs SHAP with this company's feature values (fallback)
    """
    cnum_str = company_key(row.get("company_num", ""))
    # Source 1: per-company SHAP from narrative file
    narr = narratives_dict.get(cnum_str)
    if narr:
        parsed = parse_drivers_from_narrative(narr)
        if parsed:
            drivers = [(f, s, to_float(v) if to_float(v) == to_float(v) else 0.0)
                       for f, s, v in parsed[:5]]
            return drivers, "per-company SHAP"

    # Source 2: live SHAP via TreeExplainer
    if explainer is not None and feature_cols:
        try:
            vals = [to_float(row.get(f, 0)) for f in feature_cols]
            vals = [0.0 if np.isnan(v) else v for v in vals]
            X = np.array([vals], dtype=np.float32)
            sv = explainer.shap_values(X, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[-1]
            sv = sv[0]
            abs_s = np.abs(sv)
            top_idx = abs_s.argsort()[::-1][:5]
            drivers = [(feature_cols[i], float(sv[i]),
                        to_float(row.get(feature_cols[i], 0)))
                       for i in top_idx]
            return drivers, "live SHAP (computed for this company)"
        except Exception:
            pass

    # Source 3: global SHAP fallback
    if mean_abs_shap is not None and feature_cols:
        top_idx = mean_abs_shap.argsort()[::-1][:5]
        drivers = [(feature_cols[i], float(mean_abs_shap[i]),
                    to_float(row.get(feature_cols[i], 0)))
                   for i in top_idx]
        return drivers, "global SHAP (population-level drivers + this company's values)"

    return None, "no SHAP source available"


def build_narrative_prompt(company: pd.Series, top_drivers: List[Tuple[str, float, float]],
                           score: float, base_rate: float = 0.0407) -> str:
    """Builds the prompt once - both providers use the same text."""
    drivers_txt = "\n".join(
        f"  - {name}: SHAP={shap_val:+.4f}, feature value={feat_val:.4g}"
        for name, shap_val, feat_val in top_drivers
    )
    return f"""You are an audit risk analyst writing a brief, professional narrative for an Irish SME flagged by a dissolution-risk model.

Company: {company.get('company_name', 'Unknown')}
CRO number: {company.get('company_num', '-')}
Sector: NACE {company.get('nace_v2_code', '-')}
County: {company.get('county', '-')}
Age: {to_float(company.get('company_age_years', 0)):.1f} years

Model dissolution risk score: {score:.4f}  ({score*100:.1f}% calibrated probability)
Population base rate (modelled test population): {base_rate:.2%}

Top model drivers (SHAP log-odds; positive = increases risk):
{drivers_txt}

Write exactly three sentences:
1. Headline risk in plain English, anchored on the score versus base rate.
2. Identify the two strongest behavioural signals from the drivers above. Name them and explain what the value means in business terms.
3. A concrete next step for the audit team.

Do not invent facts beyond the drivers listed. No bullet points or markdown."""


def generate_llm_narrative(provider: str, company: pd.Series,
                           top_drivers: List[Tuple[str, float, float]],
                           score: float, base_rate: float = 0.0407,
                           openai_client=None, ollama_model: str = "llama3.1:8b") -> str:
    """Route to whichever provider is configured."""
    prompt = build_narrative_prompt(company, top_drivers, score, base_rate)
    try:
        if provider == "ollama":
            return ollama_generate(ollama_model, prompt)
        elif provider == "openai" and openai_client is not None:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        else:
            return "No LLM provider configured."
    except Exception as e:
        return f"LLM call failed: {type(e).__name__}: {e}"


# =============================================================================
# STAGE 2 VALIDATION (register cross-check)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_cohort_validation() -> Optional[pd.DataFrame]:
    """nlp_05 output: which prospective companies the CRO register now shows as
    exited, on the narrower involuntary-exit rule (struck off, dissolved, or
    listed in the dissolutions file)."""
    path = OUTPUTS / "nlp" / "cohort_validation.csv"
    if not path.exists():
        return None
    keep = ("company_num", "company_name", "combined_risk_tier", "risk_score",
            "county", "any_external_confirm")
    try:
        return pd.read_csv(path, low_memory=False, usecols=lambda c: c in keep)
    except Exception:
        return None


def render_stage2_tab():
    st.markdown("#### Model Validation")
    st.caption(
        "How the risk ranking holds up against the CRO register: companies the model "
        "rates higher are far more likely to already be struck off or dissolved. This "
        "is the narrower involuntary-exit rule (struck off, dissolved, or listed in the "
        "dissolutions file), the route taken by a company that stops filing. It "
        "excludes voluntary liquidation and wind-up.")

    validation = load_cohort_validation()
    if validation is None or "any_external_confirm" not in validation.columns:
        st.info("Validation data is not available in this view.")
        return

    total = len(validation)
    has_tier = "combined_risk_tier" in validation.columns

    def rate(sub):
        if len(sub) == 0:
            return "-"
        return f"{100 * sub['any_external_confirm'].sum() / len(sub):.1f}%"

    prio = validation[validation["combined_risk_tier"] == "PRIORITY"] if has_tier else validation.iloc[0:0]
    low = validation[validation["combined_risk_tier"] == "LOW_CONCERN"] if has_tier else validation.iloc[0:0]

    # The chart below plots every tier's rate against a labelled base-rate line,
    # so any card repeating a rate here would be that chart said twice. These two
    # give the scale of the check, which the chart does not.
    c1, c2 = st.columns(2)
    c1.metric("Companies checked", f"{total:,}")
    c2.metric("Confirmed as exited", f"{int(validation['any_external_confirm'].sum()):,}",
              help="Companies the register now shows as struck off, dissolved, or "
                   "listed in the dissolutions file. How these distribute across the "
                   "tiers is the point of the check, and is plotted below.")

    if not has_tier:
        return

    st.markdown("##### Confirmation rate by risk tier")
    by_tier = validation.groupby("combined_risk_tier").agg(
        n=("company_num", "count"),
        any_external=("any_external_confirm", "sum"),
    )
    by_tier["rate"] = (100 * by_tier["any_external"] / by_tier["n"]).round(1)
    by_tier = by_tier.reset_index()
    by_tier["_o"] = by_tier["combined_risk_tier"].apply(
        lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else 99)
    by_tier = by_tier.sort_values("_o").drop(columns="_o")

    base = 100 * validation["any_external_confirm"].sum() / total
    fig = go.Figure(go.Bar(
        y=by_tier["combined_risk_tier"], x=by_tier["rate"], orientation="h",
        marker_color=[TIER_COLORS.get(t, UI_YELLOW) for t in by_tier["combined_risk_tier"]],
        text=by_tier["rate"].map("{:.1f}%".format), textposition="outside",
    ))
    fig.add_vline(x=base, line_dash="dash", line_color="#FFFFFF", line_width=1.4,
                  annotation_text=f"Cohort base rate {base:.2f}%",
                  annotation_position="top right",
                  annotation_font=dict(color=UI_TEXT_DIM, size=11))
    plotly_dark(fig, height=320, showlegend=False,
                margin=dict(t=44, b=40, l=180, r=100),
                xaxis_title="Confirmed as exited on the CRO register (%)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    disp = by_tier.rename(columns={
        "combined_risk_tier": "Risk tier", "n": "Companies",
        "any_external": "Confirmed struck off or dissolved",
    })
    st.dataframe(disp[["Risk tier", "Companies",
                       "Confirmed struck off or dissolved"]],
                 use_container_width=True, hide_index=True)
    st.caption("A lower bound. Register updates lag the event, so a company that has "
               "failed but not yet been struck off counts here as unconfirmed.")

    st.markdown("---")
    st.markdown("##### Companies the register now confirms as exited")
    st.caption("The individual companies behind the rates above. These were scored at the "
               "2024 observation date and the register has since caught up with them, so "
               "each row is a case the model ranked before the outcome was recorded.")

    conf = validation[validation["any_external_confirm"].astype(bool)].copy()
    if conf.empty:
        st.info("No confirmed exits in this cohort.")
        return

    present = [t for t in TIER_ORDER if t in conf["combined_risk_tier"].unique()]
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        opts = ["All tiers"] + [f"{t}  ({int((conf['combined_risk_tier'] == t).sum()):,})"
                                for t in present]
        pick = st.selectbox("Tier", opts, key="conf_tier",
                            label_visibility="collapsed")
    with fc2:
        name_q = st.text_input("Search", placeholder="Filter by company name or CRO number",
                               key="conf_search", label_visibility="collapsed")

    if pick != "All tiers":
        conf = conf[conf["combined_risk_tier"] == pick.split("  (")[0]]
    if name_q.strip():
        q = name_q.strip().lower()
        mask = pd.Series(False, index=conf.index)
        if "company_name" in conf.columns:
            mask |= conf["company_name"].astype(str).str.lower().str.contains(q, na=False)
        if "company_num" in conf.columns:
            mask |= conf["company_num"].astype(str).str.contains(q, na=False)
        conf = conf[mask]

    if "risk_score" in conf.columns:
        conf = conf.sort_values("risk_score", ascending=False)

    cols = [c for c in ("company_num", "company_name", "combined_risk_tier",
                        "risk_score", "county") if c in conf.columns]
    view = conf[cols].rename(columns={
        "company_num": "CRO number", "company_name": "Company",
        "combined_risk_tier": "Tier", "risk_score": "Risk score", "county": "County",
    })
    st.dataframe(
        view.head(300).style.format({"Risk score": "{:.4f}"}),
        use_container_width=True, height=400, hide_index=True)
    st.caption(f"{len(conf):,} confirmed exits"
               + (", showing first 300." if len(conf) > 300 else ".")
               + " Ranked by the score the model assigned before the outcome was known.")
    st.download_button(
        "Download confirmed exits (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="confirmed_exits.csv", mime="text/csv")


# =============================================================================
# CLIENT PORTFOLIO
# =============================================================================
def _read_client_list(upload) -> pd.DataFrame:
    """Read a csv, txt or xlsx client list into a frame."""
    name = upload.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raw = upload.getvalue().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    first = lines[0] if lines else ""
    sep = "," if "," in first else ("\t" if "\t" in first else None)
    if sep is None:
        return pd.DataFrame({"company_num": [l.strip() for l in lines if l.strip()]})
    import io as _io
    return pd.read_csv(_io.StringIO(raw), sep=sep)


def _pick_number_column(df: pd.DataFrame):
    """Find the column of CRO numbers: an obvious name, else the column with the
    highest share of purely numeric values."""
    for c in df.columns:
        if str(c).strip().lower() in ("company_num", "cro number", "cro_number", "cro",
                                      "company number", "companynumber", "number",
                                      "reg_no"):
            return c
    best, best_share = None, 0.0
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip()
        if s.empty:
            continue
        share = s.str.fullmatch(r"\d{1,7}").mean()
        if share > best_share:
            best, best_share = c, share
    return best if best_share >= 0.6 else None


def render_portfolio_tab():
    st.markdown("#### Client Portfolio Screen")
    st.caption("Upload an engagement client list to see only those companies, ranked by "
               "risk. The file is matched against the scored register in this session "
               "and is not stored.")

    if "combined_risk_tier" not in prosp.columns or "company_num" not in prosp.columns:
        st.warning("Scored register not available.")
        return

    up = st.file_uploader(
        "Client list", type=["csv", "txt", "xlsx", "xls"],
        help="Any file with a column of CRO numbers. A plain list of numbers, one per "
             "line, also works. Other columns are ignored.",
        label_visibility="collapsed")

    pasted = ""
    with st.expander("Or paste CRO numbers"):
        pasted = st.text_area("One per line", height=110, label_visibility="collapsed",
                              placeholder="628624")

    wanted, source_label = [], ""
    if up is not None:
        try:
            raw_df = _read_client_list(up)
        except Exception as e:
            st.error(f"Could not read that file: {type(e).__name__}: {e}")
            return
        col = _pick_number_column(raw_df)
        if col is None:
            st.error("No column of CRO numbers found. Columns seen: "
                     + ", ".join(str(c) for c in raw_df.columns[:12]))
            return
        wanted = [company_key(v) for v in raw_df[col].dropna()]
        source_label = f"{up.name}, column '{col}'"
    elif pasted.strip():
        wanted = [company_key(l) for l in pasted.splitlines() if l.strip()]
        source_label = "pasted list"

    if not wanted:
        st.info("Upload or paste a client list to begin.")
        return

    wanted_unique = list(dict.fromkeys(wanted))
    port = prosp[prosp["company_num"].map(company_key).isin(set(wanted_unique))].copy()
    matched = len(port)
    missing = len(wanted_unique) - matched

    m1, m2, m3 = st.columns(3)
    m1.metric("Clients in list", f"{len(wanted_unique):,}")
    m2.metric("Matched to register", f"{matched:,}")
    m3.metric("Not scored", f"{missing:,}")

    if missing:
        st.caption(
            f"{missing:,} of {len(wanted_unique):,} did not match. The scored cohort is "
            f"the {len(prosp):,} companies with a 2024 observation date, so a client "
            "that has never filed, or last filed before 2024, will not appear. Absence "
            "is not a low-risk finding.")

    if port.empty:
        st.warning(f"None of the {len(wanted_unique):,} companies in {source_label} are "
                   "in the scored cohort.")
        return

    st.markdown("---")
    counts = port["combined_risk_tier"].value_counts()
    mcols = st.columns(4)
    for col_i, tier, label in zip(mcols, TIER_ORDER,
                                  ["Priority", "Dissolution risk",
                                   "Behavioural anomaly", "Low concern"]):
        n = int(counts.get(tier, 0))
        col_i.metric(label, f"{n:,}", f"{100 * n / matched:.1f}% of portfolio",
                     delta_color="off")

    flagged = int(sum(counts.get(t, 0) for t in TIER_ORDER[:3]))
    st.caption(
        f"{flagged:,} of {matched:,} matched clients carry a signal from at least one "
        f"stage; {matched - flagged:,} clear both. The tiers are not a ladder: "
        f"DISSOLUTION_RISK holds companies in the same top-5% score band as PRIORITY "
        f"that the anomaly detector did not flag.")

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### This portfolio against the full register")
        rows = []
        for tier in TIER_ORDER:
            rows.append({
                "Tier": tier,
                "Portfolio": 100 * counts.get(tier, 0) / matched,
                "Register": 100 * (prosp["combined_risk_tier"] == tier).sum() / len(prosp),
            })
        cmp_df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=cmp_df["Tier"], x=cmp_df["Register"], orientation="h",
                             name="Full register", marker_color=UI_MID,
                             text=cmp_df["Register"].map("{:.1f}%".format),
                             textposition="outside"))
        fig.add_trace(go.Bar(y=cmp_df["Tier"], x=cmp_df["Portfolio"], orientation="h",
                             name="This portfolio", marker_color=UI_YELLOW,
                             text=cmp_df["Portfolio"].map("{:.1f}%".format),
                             textposition="outside"))
        plotly_dark(fig, height=340, barmode="group",
                    margin=dict(t=48, b=40, l=180, r=70),
                    legend=dict(orientation="h", y=1.16, x=0),
                    xaxis_title="Share of companies (%)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Whether this portfolio carries more or less risk than a random draw "
                   "from the register. Share, not count, so a small portfolio still "
                   "compares.")

    with right:
        st.markdown("##### Where the flagged clients are")
        hr = port[port["combined_risk_tier"].isin(TIER_ORDER[:3])]
        if "county" in hr.columns and not hr.empty:
            cnt = drop_unknowns(hr["county"]).value_counts().head(10)
        else:
            cnt = pd.Series(dtype=int)
        if not cnt.empty:
            cdf = cnt.reset_index()
            cdf.columns = ["County", "Flagged"]
            fig = go.Figure(go.Bar(y=cdf["County"][::-1], x=cdf["Flagged"][::-1],
                                   orientation="h", marker_color="#E07B00",
                                   text=cdf["Flagged"][::-1], textposition="outside"))
            plotly_dark(fig, height=340, showlegend=False,
                        margin=dict(t=48, b=40, l=110, r=50),
                        xaxis_title="Flagged clients")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged clients with a county in this portfolio.")

    st.markdown("---")
    st.markdown("##### Flagged clients, highest risk first")
    show_all = st.checkbox("Show every matched client, not just flagged ones",
                           value=False)
    table = port if show_all else port[port["combined_risk_tier"].isin(TIER_ORDER[:3])]
    if "dissolution_risk_score" in table.columns:
        table = table.sort_values(RANK_COL, ascending=False)

    disp = [c for c in ("company_num", "company_name", "county", "combined_risk_tier",
                        "company_age_years", "dissolution_risk_score", "anomaly_band")
            if c in table.columns]
    if "entity_type" in table.columns:
        disp = disp + ["entity_type"]

    if table.empty:
        st.success("No client in this portfolio carries a signal from either stage.")
        return

    st.dataframe(
        table[disp].head(500).style.format({
            "company_age_years": "{:.1f}",
            "dissolution_risk_score": pct_txt,
            "anomaly_band": band_txt,
        }),
        use_container_width=True, height=420)
    st.caption(f"{len(table):,} rows"
               + (", showing first 500." if len(table) > 500 else "."))

    export = table[disp].copy()
    if "dissolution_risk_score" in export.columns:
        export["dissolution_risk_score"] = (export["dissolution_risk_score"] * 100).round(1)
    if "anomaly_band" in export.columns:
        export["anomaly_band"] = (export["anomaly_band"] * 100).round(1)
    export = export.rename(columns={
        "company_num": "CRO number", "company_name": "Company", "county": "County",
        "combined_risk_tier": "Tier", "company_age_years": "Age (years)",
        "dissolution_risk_score": "Dissolution risk (%)",
        "anomaly_band": "Anomaly band (top %)",
    })
    st.download_button(
        "Download this portfolio screen (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_screen.csv", mime="text/csv",
        help="The filtered client list, for engagement planning or monitoring.")


# =============================================================================
# SECTOR PEER BENCHMARKING
# =============================================================================
# The features that carry the most signal, in the order Table 5.3 ranks them.
PEER_FEATURES = [
    ("ar_filed_count", "Annual returns filed", 0),
    ("total_submissions", "Total submissions", 0),
    ("annual_submission_rate", "Submissions per year", 2),
    ("company_age_years", "Company age (years)", 1),
    ("director_change_count", "Director changes", 0),
]

MIN_PEERS = 30


@st.cache_data(show_spinner=False)
def _peer_pool(level: str) -> Optional[pd.DataFrame]:
    """Peer groups at one of three levels of granularity.

    A four-digit NACE class is the tightest comparison but many classes hold only
    a handful of companies, and a median drawn from four peers is noise. So the
    group is widened until it holds enough companies to say something, and the
    level actually used is reported rather than hidden.
    """
    if "nace_v2_code" not in prosp.columns:
        return None
    df = prosp.copy()
    code = df["nace_v2_code"].apply(nace_digits)
    if level == "class":
        df["_grp"] = code.str[:4]
    elif level == "division":
        df["_grp"] = code.str[:2]
    else:
        df["_grp"] = "ALL"
    df = df[df["_grp"].str.len() > 0]
    return df


def _peer_group(row) -> Tuple[Optional[pd.DataFrame], str]:
    """Return the tightest peer group with at least MIN_PEERS companies, and a
    plain description of what that group is."""
    code = nace_digits(row.get("nace_v2_code", ""))
    sector = nace_section_label(row.get("nace_v2_code", "")) or "this sector"
    if not code:
        pool = _peer_pool("all")
        if pool is None or pool.empty:
            return None, ""
        return pool, (f"{len(pool):,} companies across the whole register: no sector "
                      f"is recorded against this company, so no sector comparison is "
                      f"possible")
    for level, n_digits, desc in (
        ("class", 4, f"NACE class {code[:4]}"),
        ("division", 2, f"NACE division {code[:2]} ({sector})"),
    ):
        if len(code) < n_digits:
            continue
        pool = _peer_pool(level)
        if pool is None:
            return None, ""
        grp = pool[pool["_grp"] == code[:n_digits]]
        if len(grp) >= MIN_PEERS:
            return grp, f"{len(grp):,} companies in {desc}"
    pool = _peer_pool("all")
    if pool is None or pool.empty:
        return None, ""
    return pool, (f"{len(pool):,} companies across the whole register: this company's "
                  f"sector holds too few peers to compare against")


def render_sector_benchmark(row):
    """Each key feature against its sector's spread, so an auditor can see whether
    a value is unusual for this kind of company rather than unusual in general."""
    grp, desc = _peer_group(row)
    if grp is None or grp.empty:
        st.info("Sector peers are not available for this company.")
        return

    st.caption(f"Compared against {desc}. The bar spans the middle half of the "
               f"sector, from the 25th to the 75th percentile; the line is the "
               f"sector median and the marker is this company. A value outside "
               f"the bar is unusual for its sector, which is not the same as "
               f"unusual overall.")

    rows = []
    for feat, label, dp in PEER_FEATURES:
        if feat not in grp.columns:
            continue
        v = to_float(row.get(feat))
        s = pd.to_numeric(grp[feat], errors="coerce").dropna()
        if np.isnan(v) or len(s) < 5:
            continue
        q1, med, q3 = s.quantile(0.25), s.median(), s.quantile(0.75)
        pct = 100 * (s < v).mean()
        rows.append(dict(label=label, v=v, q1=q1, med=med, q3=q3, lo=s.min(),
                         hi=s.quantile(0.99), pct=pct, dp=dp))

    if not rows:
        st.info("No comparable features for this company's sector.")
        return

    fig = go.Figure()
    for i, r in enumerate(reversed(rows)):
        y = r["label"]
        fig.add_trace(go.Scatter(
            x=[r["lo"], r["hi"]], y=[y, y], mode="lines",
            line=dict(color=UI_MID, width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[r["q1"], r["q3"]], y=[y, y], mode="lines",
            line=dict(color="#6B6B7B", width=13), hoverinfo="skip",
            showlegend=False))
        fig.add_trace(go.Scatter(
            x=[r["med"]], y=[y], mode="markers",
            marker=dict(symbol="line-ns", color="#FFFFFF", size=17,
                        line=dict(width=2, color="#FFFFFF")),
            name="Sector median", showlegend=(i == 0),
            hovertemplate=f"Sector median: {r['med']:.{r['dp']}f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[r["v"]], y=[y], mode="markers",
            marker=dict(color=UI_YELLOW, size=13, line=dict(width=1.5, color=UI_BG)),
            name="This company", showlegend=(i == 0),
            hovertemplate=(f"This company: {r['v']:.{r['dp']}f}<br>"
                           f"Higher than {r['pct']:.0f}% of the sector<extra></extra>")))
    plotly_dark(fig, height=62 * len(rows) + 96, margin=dict(t=44, b=40, l=170, r=30),
                legend=dict(orientation="h", y=1.22, x=0), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame([{
        "Feature": r["label"],
        "This company": f"{r['v']:.{r['dp']}f}",
        "Sector median": f"{r['med']:.{r['dp']}f}",
        "Sector 25th to 75th": f"{r['q1']:.{r['dp']}f} to {r['q3']:.{r['dp']}f}",
        "Percentile in sector": f"{r['pct']:.0f}th",
    } for r in rows])
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# =============================================================================
# LOAD EVERYTHING
# =============================================================================
prosp = load_prospective()
priority = load_priority()
comp_df = load_model_comparison()
meta = load_winner_meta()
narratives_dict = load_narratives_dict()
feature_cols = load_feature_cols()
shap_vals_global, mean_abs_shap = load_shap_global()

# Attach the rule-based SPV label to the prospective cohort when available.
entity_types = load_entity_types()
if entity_types is not None and prosp is not None and "company_num" in prosp.columns:
    prosp["entity_type"] = prosp["company_num"].map(
        lambda v: entity_types.get(company_key(v), "Unclassified"))
model, calibrator, explainer, winner_name, explainer_error = load_model_artefacts()


@st.cache_data(show_spinner=False)
def _rank_scores(_model, n_rows: int) -> Optional[np.ndarray]:
    """Raw classifier scores for the prospective cohort, used only to order the
    review list. The calibrated score is what gets displayed, but it is rounded
    to four places and isotonic collapses it into bands, so it holds only a few
    hundred distinct values and hundreds of companies tie. The raw score is the
    ordering every result in the write-up is computed on."""
    if _model is None or not feature_cols:
        return None
    try:
        missing = [c for c in feature_cols if c not in prosp.columns]
        if missing:
            return None
        X = prosp[feature_cols].values
        p = _model.predict_proba(X)
        return p if p.ndim == 1 else p[:, 1]
    except Exception:
        return None


# Attach the ordering key. Falls back to the calibrated score if the model or
# the feature columns are unavailable, so the app still runs.
# The anomaly band, computed once against the full cohort, so the tables and the
# metric cards express Stage 2 on the same footing as Stage 1: a position among
# companies. The raw Isolation Forest score is a min-max distance, not a
# probability, so it cannot be shown as a percentage without asserting something
# the model does not claim.
if "if_anomaly_score" in prosp.columns:
    prosp["anomaly_band"] = pd.to_numeric(
        prosp["if_anomaly_score"], errors="coerce"
    ).rank(pct=True, ascending=False, method="max")

_rank = _rank_scores(model, len(prosp))
if _rank is not None and len(_rank) == len(prosp):
    prosp["_rank_score"] = _rank
    RANK_COL = "_rank_score"
    RANK_IS_RAW = True
else:
    RANK_COL = "dissolution_risk_score" if "dissolution_risk_score" in prosp.columns else None
    RANK_IS_RAW = False

if prosp.empty:
    st.error("Company data not available...")
    st.stop()

# Sidebar - pipeline health check (helps debug missing artefacts)
with st.sidebar:
    st.markdown("### Pipeline Health")
    checks = [
        ("Prospective data",       len(prosp) > 0,                   f"{len(prosp):,} rows"),
        ("Priority companies",     len(priority) > 0,                f"{len(priority)} rows"),
        ("Winner meta",            bool(meta),                       meta.get("winner_name", "n/a")),
        ("Model comparison",       len(comp_df) > 0,                 f"{len(comp_df)} models"),
        ("Company summaries",      len(narratives_dict) > 0,         f"{len(narratives_dict)} loaded"),
        ("Feature list",           len(feature_cols) > 0,            f"{len(feature_cols)} features"),
        ("SHAP global",            mean_abs_shap is not None,        "computed"   if mean_abs_shap is not None else "missing"),
        ("Winner model loaded",    model is not None,                winner_name),
        ("SHAP explainer",         explainer is not None,
            "ready" if explainer is not None else (explainer_error or "unavailable")[:42]),
    ]
    for label, ok, detail in checks:
        icon = "✅" if ok else "⚠️"
        color = "#1A7340" if ok else "#E07B00"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;gap:8px;'>"
            f"<span style='color:{color};'>{icon} {label}</span>"
            f"<span style='color:{UI_TEXT_DIM};font-size:0.78rem;text-align:right;'>{detail}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # If explainer failed, show the full error in an expander (auto-open if broken)
    if explainer is None and explainer_error:
        with st.expander("⚠️ SHAP explainer error (full)", expanded=True):
            st.code(explainer_error, language=None)

    st.markdown("---")
    st.markdown("### LLM Provider")

    ollama_models, ollama_status = ollama_available()
    openai_client, openai_status = get_openai_client()

    ollama_ok = bool(ollama_models)
    openai_ok = openai_client is not None

    if openai_ok:
        st.markdown(f"<div style='color:#1A7340;'>OpenAI: {openai_status}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:{UI_TEXT_DIM};font-size:0.78rem;'>OpenAI: {openai_status}</div>",
                    unsafe_allow_html=True)
    if ollama_ok:
        st.markdown(f"<div style='color:#1A7340;'>Ollama (local): {ollama_status}</div>",
                    unsafe_allow_html=True)

    available = []
    if openai_ok:    available.append("OpenAI (cloud)")
    if ollama_ok:    available.append("Ollama (local, free)")
    if not available:
        available = ["None available"]

    llm_choice = st.selectbox("Active provider", available, index=0,
                              help="OpenAI runs in the cloud and uses your API credit. "
                                   "Ollama runs locally (free, no key).")

    selected_ollama_model = ollama_models[0] if ollama_models else "llama3.1:8b"
    if llm_choice.startswith("Ollama") and ollama_models:
        selected_ollama_model = st.selectbox("Ollama model", ollama_models, index=0)

    if not openai_ok:
        with st.expander("Paste OpenAI API key (optional)"):
            manual_key = st.text_input(
                "OPENAI_API_KEY", type="password", placeholder="sk-...",
                key="manual_openai_key", label_visibility="collapsed",
            )
            if manual_key:
                os.environ["OPENAI_API_KEY"] = manual_key.strip()
                st.success("Key set for this session. Reload page to pick it up.")

    if llm_choice.startswith("OpenAI"):
        active_provider = "openai"
    elif llm_choice.startswith("Ollama"):
        active_provider = "ollama"
    else:
        active_provider = "none"


# =============================================================================
# HEADER
# =============================================================================
ap_str = f"{to_float(meta.get('avg_precision', '')):.4f}" if meta.get("avg_precision") else "n/a"
auc_str = f"{to_float(meta.get('auc_roc', '')):.4f}" if meta.get("auc_roc") else "n/a"

st.markdown(f"""
<div class="{_ENTER}"><div style="background: linear-gradient(135deg, {UI_DARK} 0%, {UI_BG} 100%);
            padding: 20px 26px; border-radius: 8px;
            border-left: 4px solid {UI_YELLOW}; margin-bottom: 18px;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1 style="margin:0; font-size:1.55rem;">Irish SME Dissolution Risk Dashboard</h1>
    </div>
  </div>
</div></div>
""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
# Two audiences, one dashboard. An engagement lead wants a client list and a
# risk tier; a methodology reviewer wants the permutation null and the
# calibration. Showing both the same seven tabs serves neither well, so the
# audience picks and the tab set follows. Every tab exists in both modes; the
# mode decides which are on screen.
_mode_l, _mode_r = st.columns([2, 5])
with _mode_l:
    ui_mode = st.radio(
        "View", ["Engagement", "Methodology"], index=0, horizontal=True,
        key="ui_mode", label_visibility="collapsed",
        help="Engagement: your clients, their tier, and the evidence for one "
             "company. Methodology: how the model was selected, what drives it, "
             "and how the ranking was validated.")
with _mode_r:
    st.caption(
        "Which of your clients are at risk, why, and what the file says."
        if ui_mode == "Engagement" else
        "Model selection, feature attribution, and validation against the "
        "register.")

IS_ENGAGEMENT = ui_mode == "Engagement"

if IS_ENGAGEMENT:
    tab1, tab7, tab3 = st.tabs([
        "Executive Overview",
        "Client Portfolio",
        "Company Lookup",
    ])
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Overview",
        "Risk Tiers",
        "Company Lookup",
        "Model Performance",
        "Risk Factors",
        "Model Validation",
    ])


# ----------------------------------------------------------------------------
# TAB 1 - EXECUTIVE OVERVIEW
# ----------------------------------------------------------------------------
with tab1:
    if "combined_risk_tier" in prosp.columns:
        tier_counts = prosp["combined_risk_tier"].value_counts()
    else:
        tier_counts = pd.Series(dtype=int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Priority", f"{tier_counts.get('PRIORITY', 0):,}",
              help="Top 5% supervised + Isolation Forest anomaly")
    c2.metric("Dissolution Risk", f"{tier_counts.get('DISSOLUTION_RISK', 0):,}")
    c3.metric("Behavioural Anomaly", f"{tier_counts.get('BEHAVIORAL_ANOMALY', 0):,}")
    c4.metric("Low Concern", f"{tier_counts.get('LOW_CONCERN', 0):,}")
    c5.metric("Total Scored", f"{len(prosp):,}")

    st.markdown("")
    st.markdown("##### From the full register to a focused review list")
    if not tier_counts.empty:
        tdf = tier_counts.reset_index()
        tdf.columns = ["Tier", "Count"]
        tdf = tdf.sort_values("Count", ascending=True)
        grand = int(tdf["Count"].sum())
        tdf["Share"] = (100 * tdf["Count"] / grand).round(2)
        # The cards above already carry the counts. What they cannot show is the
        # shape: four tiers spanning three orders of magnitude. So the bars carry
        # the share and nothing else.
        tdf["Label"] = tdf["Share"].astype(str) + "%"
        # One trace with a colour list, not px.bar(color=...). Colouring by a
        # column makes Plotly build one trace per colour, and four traces across
        # four categories each get their own offset slot inside the category
        # band, which is what pushed every bar off its own label.
        fig = go.Figure(go.Bar(
            y=tdf["Tier"], x=tdf["Count"], orientation="h",
            marker_color=[TIER_COLORS.get(t, UI_YELLOW) for t in tdf["Tier"]],
            text=tdf["Label"], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:,} companies<extra></extra>",
        ))
        plotly_dark(fig, height=320, showlegend=False,
                    margin=dict(t=20, b=44, l=180, r=110),
                    xaxis_title="Companies (log scale)",
                    # Plotly's default log axis prints every minor tick, so the
                    # labels read 6 7 8 9 100 2 3 4 5 6 7 8 9 1000. Only the
                    # decades mean anything here.
                    xaxis=dict(type="log", tickmode="array",
                               tickvals=[10, 100, 1000, 10000],
                               ticktext=["10", "100", "1,000", "10,000"],
                               range=[1, 4.6]),
                    bargap=0.35)
        st.plotly_chart(fig, use_container_width=True)
        n_flagged = int(tier_counts.get("PRIORITY", 0) + tier_counts.get("DISSOLUTION_RISK", 0)
                        + tier_counts.get("BEHAVIORAL_ANOMALY", 0))
        n_prio = int(tier_counts.get("PRIORITY", 0))
        _prio_share = 100 * n_prio / len(prosp) if len(prosp) else 0
        st.caption(f"The tiers span three orders of magnitude, so the axis is log "
                   f"scaled to keep the small high-risk tiers visible. Companies where "
                   f"both stages agree make up {_prio_share:.2f}% of the population.")

    st.markdown("---")
    cC, cD = st.columns(2)

    with cC:
        st.markdown("##### Geographic Risk Concentration")
        map_rendered = False
        if "county" in prosp.columns and "combined_risk_tier" in prosp.columns:
            hr = prosp[prosp["combined_risk_tier"].isin(["PRIORITY", "DISSOLUTION_RISK"])]
            county_counts = drop_unknowns(hr["county"]).value_counts().reset_index()
            county_counts.columns = ["county", "flagged"]

            def _norm(s):
                s = str(s).lower().strip()
                for p in ("co. ", "county ", "co "):
                    if s.startswith(p):
                        s = s[len(p):]
                s = (s.replace(" city", "").replace(" county", "")
                       .replace("north ", "").replace("south ", "")
                       .replace("county", "").strip())
                return s

            county_counts["_key"] = county_counts["county"].apply(_norm)

            geojson = None
            # Prefer a local geojson file (most reliable); fall back to URLs.
            local_geo = ROOT / "ireland_counties.geojson"
            if local_geo.exists():
                try:
                    import json as _json
                    geojson = _json.loads(local_geo.read_text(encoding="utf-8"))
                except Exception:
                    geojson = None
            if not (geojson and geojson.get("features")):
                for geo_url in (
                    "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/ireland.geojson",
                    "https://raw.githubusercontent.com/mrcagney/ireland_geojson/master/counties.geojson",
                ):
                    try:
                        import json as _json
                        import urllib.request as _url
                        with _url.urlopen(geo_url, timeout=8) as resp:
                            geojson = _json.load(resp)
                        if geojson.get("features"):
                            break
                    except Exception:
                        geojson = None

            if geojson and geojson.get("features"):
                # Detect the property key holding county names, add a normalized key.
                props0 = geojson["features"][0].get("properties", {})
                key = None
                for cand in ("name", "NAME", "county", "COUNTY", "NAME_TAG",
                             "NAME_EN", "CountyName", "id"):
                    if cand in props0:
                        key = cand
                        break
                if key:
                    for feat in geojson["features"]:
                        nm = feat.get("properties", {}).get(key, "")
                        feat["properties"]["_key"] = _norm(nm)
                    try:
                        fig = px.choropleth(
                            county_counts, geojson=geojson, locations="_key",
                            featureidkey="properties._key", color="flagged",
                            color_continuous_scale=["#1A7340", "#E07B00", "#C1121F"],
                            custom_data=["county", "flagged"], scope="europe",
                        )
                        fig.update_traces(
                            hovertemplate="<b>%{customdata[0]}</b><br>"
                                          "%{customdata[1]:,} flagged companies"
                                          "<extra></extra>")
                        fig.update_geos(fitbounds="locations", visible=False)
                        fig.update_layout(height=460, margin=dict(t=10, b=10, l=10, r=10),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          geo_bgcolor="rgba(0,0,0,0)",
                                          coloraxis_colorbar=dict(title="Flagged"))
                        st.plotly_chart(fig, use_container_width=True)
                        map_rendered = True
                    except Exception:
                        map_rendered = False

        if not map_rendered:
            st.markdown("##### High-Risk Companies by County (share of flagged)")
            if "county" in prosp.columns and "combined_risk_tier" in prosp.columns:
                hr = prosp[prosp["combined_risk_tier"].isin(["PRIORITY", "DISSOLUTION_RISK"])]
                labelled = drop_unknowns(hr["county"])
                total = len(labelled)
                cnt = labelled.value_counts().head(15).reset_index()
                cnt.columns = ["County", "Count"]
                if not cnt.empty and total:
                    cnt["Share"] = (100 * cnt["Count"] / total).round(1)
                    cnt["Label"] = cnt["Share"].astype(str) + "%  (" + cnt["Count"].astype(str) + ")"
                    fig = px.bar(cnt.iloc[::-1], x="Share", y="County", orientation="h",
                                 text="Label", color_discrete_sequence=[UI_YELLOW])
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    plotly_dark(fig, height=440, margin=dict(t=20, b=40, l=110, r=60),
                                xaxis_title="Share of flagged companies (%)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No counties to display.")

    with cD:
        st.markdown("##### High-Risk Companies by Sector (share of flagged)")
        prosp_sec = add_sector_label(prosp)
        if "sector_label" in prosp_sec.columns:
            hr = prosp_sec[prosp_sec["combined_risk_tier"].isin(["PRIORITY", "DISSOLUTION_RISK"])]
            labelled = drop_unknowns(hr["sector_label"])
            total = len(labelled)
            cnt = labelled.value_counts().head(12).reset_index()
            cnt.columns = ["Sector", "Count"]
            if not cnt.empty and total:
                cnt["Share"] = (100 * cnt["Count"] / total).round(1)
                cnt["Label"] = cnt["Share"].astype(str) + "%  (" + cnt["Count"].astype(str) + ")"
                fig = px.bar(cnt.iloc[::-1], x="Share", y="Sector", orientation="h",
                             text="Label", color_discrete_sequence=["#E07B00"])
                fig.update_traces(textposition="outside", cliponaxis=False)
                plotly_dark(fig, height=440, margin=dict(t=20, b=40, l=200, r=60),
                            xaxis_title="Share of flagged companies (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sectors to display.")


# ----------------------------------------------------------------------------
# TAB 2 - RISK TIERS (all four tiers selectable)
# ----------------------------------------------------------------------------
if not IS_ENGAGEMENT:
    with tab2:
        if "combined_risk_tier" not in prosp.columns:
            st.warning("Risk tier data not available.")
        else:
            tier_counts = prosp["combined_risk_tier"].value_counts()

            # Tier picker - default to PRIORITY since it's the most actionable
            tier_options = ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]
            tier_options = [t for t in tier_options if t in tier_counts.index]
            tier_labels = [f"{t}  ({tier_counts.get(t, 0):,})" for t in tier_options]

            # The dropdown already states the tier and its count, so a card beside it
            # saying the same two things is furniture. The definition is what a reader
            # cannot get from the dropdown, so that is all that goes underneath it.
            sel_label = st.selectbox("Risk tier", tier_labels, index=0,
                                     label_visibility="collapsed")
            sel_tier = sel_label.split("  (")[0]
            tier_color = TIER_COLORS.get(sel_tier, UI_YELLOW)
            st.caption(TIER_DESCRIPTIONS.get(sel_tier, ""))

            # Source the rows: PRIORITY → use priority df (has all columns); else use prosp
            if sel_tier == "PRIORITY" and not priority.empty:
                tier_df = priority.copy()
            else:
                tier_df = prosp[prosp["combined_risk_tier"] == sel_tier].copy()

            # step5_priority_companies.csv is a separate file, so it carries neither the
            # entity label nor the ordering key. Map both across by company number, and
            # fall back to the calibrated score if the key is unavailable.
            _sort_col = RANK_COL
            if RANK_COL == "_rank_score" and "company_num" in tier_df.columns:
                if "_rank_score" not in tier_df.columns:
                    _rk = dict(zip(prosp["company_num"].map(company_key),
                                   prosp["_rank_score"]))
                    tier_df["_rank_score"] = tier_df["company_num"].map(
                        lambda v: _rk.get(company_key(v), np.nan))
                if tier_df["_rank_score"].isna().all():
                    _sort_col = "dissolution_risk_score"
            if _sort_col not in tier_df.columns:
                _sort_col = "dissolution_risk_score" if "dissolution_risk_score" in tier_df.columns else None

            # Ensure the SPV label is present regardless of which source built tier_df.
            if (entity_types is not None and "company_num" in tier_df.columns
                    and "entity_type" not in tier_df.columns):
                tier_df["entity_type"] = tier_df["company_num"].map(
                    lambda v: entity_types.get(company_key(v), "Unclassified"))

            st.markdown("")

            # Filters
            f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
            with f1:
                counties = (["All"] + sorted(drop_unknowns(tier_df["county"]).unique().tolist())
                            if "county" in tier_df.columns else ["All"])
                sel_county = st.selectbox("County", counties, key=f"county_{sel_tier}")
            with f2:
                if "nace_v2_code" in tier_df.columns:
                    tier_df = add_sector_label(tier_df)
                    sectors = ["All"] + sorted(drop_unknowns(tier_df["sector_label"]).unique().tolist())
                    sel_sector = st.selectbox("Sector", sectors, key=f"sector_{sel_tier}")
                else:
                    sel_sector = "All"
            with f3:
                if "dissolution_risk_score" in tier_df.columns:
                    # The table this filters shows percentages, so the filter shows
                    # percentages. A slider reading 0.50 beside a column reading 59.6%
                    # invites the reader to think they are different quantities.
                    min_pct = st.slider("Minimum dissolution risk", 0, 100, 0, 5,
                                        format="%d%%", key=f"score_{sel_tier}")
                    min_score = min_pct / 100.0
                else:
                    min_score = 0.0
            with f4:
                if "entity_type" in tier_df.columns:
                    _counts = tier_df["entity_type"].value_counts()
                    _opts = ["All companies"] + [
                        f"{lbl}  ({int(_counts.get(lbl, 0)):,})"
                        for lbl in ("Special purpose vehicle", "Holding company",
                                    "Trading business", "Unclassified")
                        if _counts.get(lbl, 0) > 0]
                    _pick = st.selectbox(
                        "Entity type", _opts, key=f"ent_{sel_tier}",
                        help="A special purpose vehicle is wound up on schedule once its "
                             "purpose is served, so its exit is not distress. Unclassified "
                             "means the company name carried no signal either way, not that "
                             "the classification failed.")
                    sel_ent = "All companies" if _pick == "All companies" else _pick.split("  (")[0]
                else:
                    sel_ent = "All companies"

            filt = tier_df.copy()
            if sel_county != "All" and "county" in filt.columns:
                filt = filt[filt["county"] == sel_county]
            if sel_sector != "All" and "sector_label" in filt.columns:
                filt = filt[filt["sector_label"] == sel_sector]
            if "dissolution_risk_score" in filt.columns:
                filt = filt[filt["dissolution_risk_score"].fillna(0) >= min_score]
            if sel_ent != "All companies" and "entity_type" in filt.columns:
                filt = filt[filt["entity_type"] == sel_ent]

            # Sort by score so the worst are at the top
            if _sort_col:
                filt = filt.sort_values(_sort_col, ascending=False)

            st.markdown(f"**{len(filt):,} companies match filters** "
                        f"(showing first {min(200, len(filt))})")

            display_cols = [c for c in ["company_num", "company_name", "county",
                                        "company_age_years", "dissolution_risk_score",
                                        "anomaly_band"]
                            if c in filt.columns]
            if "entity_type" in filt.columns:
                display_cols = display_cols + ["entity_type"]
            st.dataframe(
                filt[display_cols].head(200).style.format({
                    "company_age_years": "{:.1f}",
                    "dissolution_risk_score": pct_txt,
                    "anomaly_band": band_txt,
                }),
                use_container_width=True,
                height=420,
            )

            # Exportable watchlist: the auditor's take-away work product.
            export_df = filt[display_cols].copy()
            if "dissolution_risk_score" in export_df.columns:
                export_df["dissolution_risk_score"] = (
                    export_df["dissolution_risk_score"] * 100).round(1)
            if "anomaly_band" in export_df.columns:
                export_df["anomaly_band"] = (export_df["anomaly_band"] * 100).round(1)
            export_df = export_df.rename(columns={
                "company_num": "CRO number", "company_name": "Company",
                "county": "County", "company_age_years": "Age (years)",
                "dissolution_risk_score": "Dissolution risk (%)",
                "anomaly_band": "Anomaly band (top %)",
                "entity_type": "Entity type",
            })
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download this watchlist (CSV)",
                data=csv_bytes,
                file_name=f"watchlist_{sel_tier.lower()}.csv",
                mime="text/csv",
                help="Export the filtered company list for monitoring or audit planning.",
            )

            with st.expander("Analytics platform feed (all scored companies)"):
                st.caption("A flat file matched on company_num carrying the tier, the risk "
                           "score, the top five contributing features with their signed "
                           "contributions, and the model's validation statistics as "
                           "provenance columns. Built so the scores can join an existing "
                           "anomaly-detection layer rather than requiring a parallel "
                           "platform.")
                _helix_py = ROOT / "build_helix_export.py"
                if not _helix_py.exists():
                    st.caption(f"build_helix_export.py not found next to {_THIS_FILE}.")
                elif st.button("Build platform feed (CSV)", key="genhelix"):
                    try:
                        from build_helix_export import build_helix_export
                        _sp = OUTPUTS / "prospective_shap.csv"
                        if not _sp.exists():
                            st.caption("Needs outputs/prospective_shap.csv.")
                        else:
                            _fp = DATA_PROC / "prospective_final.csv"
                            with st.spinner("Building feed…"):
                                _hx = build_helix_export(_sp, _fp if _fp.exists() else None)
                            st.download_button(
                                "Download platform feed (CSV)",
                                data=_hx.to_csv(index=False).encode("utf-8"),
                                file_name="helix_export.csv", mime="text/csv",
                                key="dlhelix")
                            st.caption(f"{len(_hx):,} rows, {len(_hx.columns)} columns.")
                    except Exception as _e:
                        st.caption(f"Could not build the feed ({type(_e).__name__}: {_e}).")

            st.markdown("---")
            colF1, colF2 = st.columns(2)

            with colF1:
                st.markdown("##### How These Companies Were Flagged")
                if {"dissolution_risk_score", "if_anomaly_score"}.issubset(filt.columns) and not filt.empty:
                    hi_risk = filt["dissolution_risk_score"] >= 0.5
                    hi_anom = filt["if_anomaly_score"] >= 0.5
                    both = int((hi_risk & hi_anom).sum())
                    risk_only = int((hi_risk & ~hi_anom).sum())
                    anom_only = int((~hi_risk & hi_anom).sum())
                    neither = int((~hi_risk & ~hi_anom).sum())
                    cats = pd.DataFrame({
                        "Flag source": [
                            "Both models (highest priority)",
                            "Dissolution risk only (Stage 1)",
                            "Anomaly only (Stage 2)",
                            "Lower signal",
                        ],
                        "Companies": [both, risk_only, anom_only, neither],
                    })
                    cats = cats[cats["Companies"] > 0]
                    cmap = {
                        "Both models (highest priority)": "#C1121F",
                        "Dissolution risk only (Stage 1)": "#E07B00",
                        "Anomaly only (Stage 2)": "#2E86DE",
                        "Lower signal": "#1A7340",
                    }
                    fig = go.Figure(go.Pie(
                        labels=cats["Flag source"], values=cats["Companies"], hole=0.55,
                        marker=dict(colors=[cmap.get(x, "#888") for x in cats["Flag source"]]),
                        textinfo="label+percent", textposition="outside",
                        textfont=dict(size=12, color="#FFFFFF"), sort=False,
                    ))
                    fig.update_layout(
                        height=340, margin=dict(t=30, b=30, l=30, r=30),
                        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#FFFFFF"},
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with colF2:
                st.markdown("##### Risk Score Spread")
                if "dissolution_risk_score" in filt.columns and not filt.empty:
                    _s = pd.to_numeric(filt["dissolution_risk_score"],
                                       errors="coerce").dropna()
                    box = go.Figure(go.Box(
                        y=_s, name=sel_tier,
                        marker_color="#4EA8DE", line_color="#4EA8DE",
                        fillcolor="rgba(78,168,222,0.25)", boxmean=True,
                        boxpoints="outliers",
                        hoverinfo="skip",
                    ))
                    # Plotly prints eight statistics on hover and nobody hovers during
                    # a demo, so the same eight are pinned beside the box. The fences
                    # are not decorative: they are where the whiskers stop, and any
                    # point beyond one is drawn as an outlier. They are computed the way
                    # Plotly computes them, as the most extreme observation still inside
                    # 1.5 x IQR, which is why a fence equal to the max means there are
                    # no outliers above it.
                    _H = 420
                    if len(_s):
                        _q1, _q3 = _s.quantile(0.25), _s.quantile(0.75)
                        _iqr = _q3 - _q1
                        _uf = _s[_s <= _q3 + 1.5 * _iqr].max()
                        _lf = _s[_s >= _q1 - 1.5 * _iqr].min()
                        _stats = [("max", _s.max()), ("upper fence", _uf),
                                  ("q3", _q3), ("median", _s.median()),
                                  ("mean", _s.mean()), ("q1", _q1),
                                  ("lower fence", _lf), ("min", _s.min())]

                        # Labels are placed at their own value, then pushed down only
                        # far enough to clear the one above. Two of these commonly share
                        # a value exactly (max with the upper fence when there are no
                        # outliers above, min with the lower fence when there are none
                        # below), so without this they would print on top of each other.
                        _lo, _hi = float(_s.min()), float(_s.max())
                        _pad = max((_hi - _lo) * 0.06, 0.01)
                        _y0, _y1 = _lo - _pad, _hi + _pad
                        _plot_px = _H - 46
                        _MIN_GAP = 23
                        _used = []
                        for _k, _v in sorted(_stats, key=lambda t: -t[1]):
                            _true_px = (1 - (_v - _y0) / (_y1 - _y0)) * _plot_px
                            _px = _true_px
                            while any(abs(_px - u) < _MIN_GAP for u in _used):
                                _px += _MIN_GAP
                            _used.append(_px)
                            box.add_annotation(
                                x=0.60, xref="paper", y=_v, yref="y",
                                yshift=-(_px - _true_px),
                                text=f"({sel_tier}, {_k}: {_v:.1%})",
                                showarrow=True, arrowhead=0, arrowwidth=1,
                                arrowcolor="rgba(255,230,0,.55)", ax=-26, ay=0,
                                xanchor="left",
                                bgcolor="rgba(26,26,35,.92)", bordercolor=UI_YELLOW,
                                borderwidth=1, borderpad=5,
                                font=dict(family="IBM Plex Mono, monospace", size=10.5,
                                          color="#FFFFFF"))
                        plotly_dark(box, height=_H, showlegend=False,
                                    margin=dict(t=18, b=28, l=52, r=26),
                                    yaxis_title="Dissolution risk (calibrated probability)",
                                    yaxis=dict(tickformat=".0%", range=[_y0, _y1]))
                    else:
                        plotly_dark(box, height=_H, showlegend=False,
                                    margin=dict(t=18, b=28, l=52, r=26),
                                    yaxis_title="Dissolution risk (calibrated probability)",
                                    yaxis=dict(tickformat=".0%"))
                    st.plotly_chart(box, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 3 - COMPANY LOOKUP (with live LLM narrative)
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("#### Company Lookup")
    st.caption("Search by CRO number or company name to see the risk assessment, key drivers, and summary.")

    search = st.text_input("Search", placeholder="e.g. 628624 or Acme Limited",
                           label_visibility="collapsed")

    if not search:
        st.markdown("**Top 20 highest-risk companies**")
        st.caption("Ranked by the model score. Search a CRO number or company name "
                   "above for the full assessment on any one of them.")
        if "dissolution_risk_score" in prosp.columns:
            top20 = prosp.nlargest(20, RANK_COL).copy()
            # No score column. The calibrated scale saturates, so the top of this
            # list is a block of identical 100.0% values that separate nothing and
            # read as either a broken model or a claim of certainty. The rank is
            # the fact this table exists to give; the score is on the company's own
            # page, where the gauge and its caveats can carry it properly.
            top20.insert(0, "rank", range(1, len(top20) + 1))
            disp = [c for c in ("rank", "company_num", "company_name", "county",
                                "anomaly_band", "combined_risk_tier")
                    if c in top20.columns]
            st.dataframe(
                top20[disp].rename(columns={
                    "rank": "Rank", "company_num": "CRO number",
                    "company_name": "Company", "county": "County",
                    "anomaly_band": "Anomaly band",
                    "combined_risk_tier": "Tier"}).style.format({
                        "Anomaly band": band_txt,
                    }),
                use_container_width=True, height=460, hide_index=True,
            )
    else:
        if search.strip().isdigit():
            match = prosp[prosp["company_num"].astype(str) == search.strip()]
        else:
            match = prosp[prosp["company_name"].astype(str)
                          .str.contains(search.strip(), case=False, na=False)].head(10)

        if match.empty:
            st.warning(f"No company matches '{search}'")
        else:
            if len(match) > 1:
                opts = match.apply(
                    lambda r: f"{r.get('company_num','?')} | {r.get('company_name','?')[:60]}", axis=1
                ).tolist()
                pick = st.selectbox("Multiple matches, choose one", opts)
                cnum = pick.split(" | ")[0]
                row = match[match["company_num"].astype(str) == cnum].iloc[0]
            else:
                row = match.iloc[0]

            # Top metrics
            risk = to_float(row.get("dissolution_risk_score", 0))
            anom = to_float(row.get("if_anomaly_score", 0))
            tier = str(row.get("combined_risk_tier", "n/a"))

            # The gauge below is where the calibrated probability lives: it is
            # larger, and it carries the colour context a card cannot. Repeating
            # the same figure here would put it on the screen twice.
            _a_all = (pd.to_numeric(prosp["if_anomaly_score"], errors="coerce").dropna()
                      if "if_anomaly_score" in prosp.columns else pd.Series(dtype=float))
            _a_band = (f"Top {100 * int((_a_all >= anom).sum()) / len(_a_all):.1f}%"
                       if (len(_a_all) and not np.isnan(anom)) else "n/a")
            c1, c2 = st.columns(2)
            c1.metric("Tier", tier, help=TIER_DESCRIPTIONS.get(tier, ""))
            c2.metric("Anomaly band", _a_band,
                      help="How unusual this company's filing pattern is, as a position "
                           "among all scored companies. The underlying Isolation Forest "
                           "score is a relative distance from the typical filing "
                           "pattern, not a probability, so it is shown as a band.")

            # AI second opinion (nlp_09): an independent LLM classifies the same
            # company from behavioural features alone, with the model's tier/score
            # withheld. Agreement is corroboration; divergence is a triage flag.
            _mvl_dict = load_model_vs_llm()
            if _mvl_dict is not None:
                _llm_tier = _mvl_dict.get(company_key(row.get("company_num", "")))
                if _llm_tier and _llm_tier.lower() != "nan":
                    _elev = {"PRIORITY", "DISSOLUTION_RISK"}
                    _model_elev = tier in _elev
                    _llm_elev = _llm_tier.upper() in _elev
                    if _model_elev == _llm_elev:
                        _txt, _col = (f"Independent AI review agrees: both flag this "
                                      f"company as {'elevated' if _model_elev else 'low'} "
                                      f"concern (AI tier: {_llm_tier})."), "#1A7340"
                    else:
                        _txt, _col = (f"Independent AI review differs: the model rates this "
                                      f"{'elevated' if _model_elev else 'low'} while the AI "
                                      f"rates it {'elevated' if _llm_elev else 'low'} "
                                      f"(AI tier: {_llm_tier}). Worth a second look."), "#E07B00"
                    st.markdown(
                        f"<div style='background:{UI_DARK};padding:10px 14px;border-radius:6px;"
                        f"border-left:3px solid {_col};margin-top:6px;'>"
                        f"<span style='color:{_col};font-weight:600;'>AI second opinion</span>"
                        f"<span style='color:{UI_TEXT_DIM};font-size:0.85rem;margin-left:10px;'>"
                        f"{_txt}</span></div>", unsafe_allow_html=True)

            st.markdown("---")
            cA, cB = st.columns([1, 1])

            with cA:
                st.markdown("##### Company Profile")
                profile = [
                    ("CRO Number", row.get("company_num", "n/a")),
                    ("Name", row.get("company_name", "n/a")),
                    ("County", row.get("county", "n/a")),
                    ("NACE Code", row.get("nace_v2_code", "n/a")),
                    ("Sector", nace_section_label(row.get("nace_v2_code", "")) or "n/a"),
                    ("Company Type", row.get("company_type", "n/a")),
                    ("Registration Date", row.get("comp_reg_date", "n/a")),
                    ("Last AR Date", row.get("last_ar_date", "n/a")),
                    ("Age (years)", f"{to_float(row.get('company_age_years', 0)):.1f}"),
                    ("AR Filed Count", safe_int(row.get("ar_filed_count", 0))),
                    ("Total Submissions", safe_int(row.get("total_submissions", 0))),
                ]
                for k, v in profile:
                    st.markdown(f"<div class='profile-row'><span>{k}</span><span>{v}</span></div>",
                                unsafe_allow_html=True)

            with cB:
                # Where this company sits in the scored cohort by Stage 1 score.
                # A rank is meaningful at the ceiling where a probability is not.
                # A percentile band, not a rank position: the calibrated score has
                # heavy ties, so hundreds of companies can share the top value and
                # "1 of 28,974" would be true of all of them.
                # The anomaly score is a min-max position on the Isolation Forest
                # scale, not a probability, so it cannot be shown as a percentage.
                # Expressed as a band it sits on the same footing as the Stage 1
                # band beside it and cannot be misread as "93% likely anomalous".
                _anom_txt, _anom_help = "n/a", ""
                if not np.isnan(anom) and "if_anomaly_score" in prosp.columns:
                    _a = pd.to_numeric(prosp["if_anomaly_score"], errors="coerce").dropna()
                    if len(_a):
                        _n_ge = int((_a >= anom).sum())
                        _anom_txt = f"Top {100 * _n_ge / len(_a):.1f}%"
                        _anom_help = (
                            f"{_n_ge:,} of {len(_a):,} companies have filing patterns at "
                            f"least as unusual as this one. The underlying Isolation "
                            f"Forest score is {anom:.2f} on a 0 to 1 scale; it measures "
                            f"how far this company sits from the typical filing pattern, "
                            f"and is not a probability, so it is shown as a band rather "
                            f"than a percentage.")

                _rank_txt, _rank_help = "n/a", ""
                if not np.isnan(risk) and "dissolution_risk_score" in prosp.columns:
                    _all = prosp["dissolution_risk_score"].dropna()
                    if len(_all):
                        _n_at_or_above = int((_all >= risk).sum())
                        _band = 100 * _n_at_or_above / len(_all)
                        _rank_txt = f"Top {_band:.1f}%"
                        _rank_help = (f"{_n_at_or_above:,} of {len(_all):,} scored "
                                      f"companies score at or above this one. The "
                                      f"calibrated score has heavy ties, so this is a "
                                      f"band rather than a position.")

                st.markdown("##### Calibrated dissolution probability")
                score_val = (risk if not np.isnan(risk) else 0) * 100
                _saturated = score_val >= 99.95
                _floored = score_val <= 0.05
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_val,
                    number={"suffix": "%", "valueformat": ".1f",
                            "font": {"size": 40, "color": "#FFFFFF"}},
                    title={"text": pct_txt(risk) if score_val < 0.1 else "",
                           "font": {"size": 13, "color": UI_TEXT_DIM}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": UI_TEXT_DIM,
                                 "tickfont": {"color": UI_TEXT_DIM, "size": 10}},
                        "bar": {"color": "rgba(0,0,0,0)"},
                        "steps": [
                            {"range": [0, 40], "color": "#1A7340"},
                            {"range": [40, 70], "color": "#E07B00"},
                            {"range": [70, 100], "color": "#C1121F"},
                        ],
                        "threshold": {
                            "line": {"color": "#FFFFFF", "width": 4},
                            "thickness": 0.85, "value": score_val,
                        },
                    },
                ))
                gauge.update_layout(
                    height=240, margin=dict(t=10, b=10, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#FFFFFF"},
                )
                st.plotly_chart(gauge, use_container_width=True)

                base_rate = 0.0407
                rel = risk / base_rate if (not np.isnan(risk) and base_rate > 0) else 0
                if _saturated:
                    _rel_txt = f"up to {rel:.1f}\u00d7"
                elif _floored:
                    _rel_txt = "below the measurable floor of"
                else:
                    _rel_txt = f"{rel:.1f}\u00d7"
                # Isotonic calibration is bounded by the rate observed in each
                # score band, so both ends of the scale saturate: 0.0 and 1.0 are
                # the endpoints of the method, not statements of certainty.
                if _saturated or _floored:
                    _end, _why = (
                        ("Top of the calibrated scale",
                         "every company in the highest score band dissolved, so the "
                         "calibration returns its upper bound. Read this as the top of "
                         "the scale, not as certainty that this company will dissolve.")
                        if _saturated else
                        ("Bottom of the calibrated scale",
                         "no company in the lowest score band dissolved, so the "
                         "calibration returns its lower bound. Read this as the bottom "
                         "of the scale, not as evidence that this company will not "
                         "dissolve. Absence of signal is not clearance.")
                    )
                    st.markdown(
                        f"<div style='background:{UI_DARK};padding:10px 14px;"
                        f"border-radius:6px;border-left:3px solid #E07B00;"
                        f"margin-top:8px;'>"
                        f"<span style='color:#E07B00;font-weight:600;'>{_end}</span>"
                        f"<span style='color:{UI_TEXT_DIM};font-size:0.85rem;"
                        f"margin-left:10px;'>Isotonic calibration is bounded by the "
                        f"dissolution rate observed in each score band: {_why}</span>"
                        f"</div>", unsafe_allow_html=True)

                # The gauge above already carries the probability and the card at
                # the top of the page already carries the anomaly band. Repeating
                # either here fills the panel without adding a fact. What is not
                # said anywhere else is where the company ranks, and how its risk
                # compares with the population, so those are what this row shows.
                mc1, mc2 = st.columns(2)
                mc1.metric("Stage 1 risk band", _rank_txt,
                           help=_rank_help or "Where this company sits among all scored "
                                              "companies by calibrated dissolution risk.")
                mc2.metric("Versus population", f"{_rel_txt} base rate",
                           help=f"The calibrated probability against the {base_rate:.2%} "
                                f"dissolution rate of the modelled population. A company "
                                f"at the top of the calibrated scale is shown as 'up to', "
                                f"because multiplying out from a saturated ceiling would "
                                f"assert a precision the calibration does not carry.")

            st.markdown("---")

            # Pull drivers from whichever source is best available
            drivers, drivers_source = get_drivers_for_company(
                row, narratives_dict, explainer, feature_cols, mean_abs_shap
            )

            st.markdown("##### What is Driving This Company's Risk")
            if drivers:
                top_df = pd.DataFrame({
                    "Feature": [feature_label(d[0]) for d in drivers],
                    "SHAP": [d[1] for d in drivers],
                    "Value": [d[2] for d in drivers],
                })
                fig = px.bar(top_df.iloc[::-1], x="SHAP", y="Feature", orientation="h",
                             color="SHAP",
                             color_continuous_scale=["#1A7340", "#FFFFFF", "#C1121F"],
                             color_continuous_midpoint=0,
                             hover_data={"Value": ":.3f"})
                plotly_dark(fig, height=320,
                            margin=dict(t=10, b=40, l=240, r=20),
                            coloraxis_showscale=False,
                            xaxis_title="Influence on risk  (right = raises risk, left = lowers risk)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk-driver detail is not available for this company.")

            st.markdown("---")
            st.markdown("##### How This Company Compares With Its Sector")
            with safe_block("Sector benchmark"):
                render_sector_benchmark(row)

            # ── LLM Audit Narrative (always shown, decoupled from live SHAP) ──
            st.markdown("---")
            st.markdown("##### Summary")
            cnum_str = company_key(row.get("company_num", ""))
            static_narr = narratives_dict.get(cnum_str)

            # Saved summary shown by default; live regeneration is a secondary option.
            if static_narr:
                st.markdown(f"<div class='narrative-box'>{static_narr}</div>",
                            unsafe_allow_html=True)

            if active_provider != "none" and drivers is not None:
                with st.expander("Regenerate summary"):
                    do_live = st.button("Regenerate", use_container_width=True)
                    if do_live:
                        spinner_msg = ("Generating (first call may take 20s)…"
                                       if active_provider == "ollama"
                                       else "Generating…")
                        with st.spinner(spinner_msg):
                            narr = generate_llm_narrative(
                                provider=active_provider,
                                company=row,
                                top_drivers=drivers,
                                score=risk if not np.isnan(risk) else 0,
                                openai_client=openai_client,
                                ollama_model=selected_ollama_model,
                            )
                        st.markdown(f"<div class='narrative-box'>{narr}</div>",
                                    unsafe_allow_html=True)
            elif not static_narr:
                st.info("No summary available for this company.")

            # ── Per-company audit report (filed-evidence PDF) ──
            st.markdown("---")
            st.markdown("##### Audit report")
            if not _REPORT_OK:
                st.caption("Report generator not available. Put per_company_report.py "
                           f"next to {_THIS_FILE} and run: "
                           "pip install reportlab matplotlib")
                st.caption(f"(reason: {_REPORT_ERR})")
            else:
                _cnum = row.get("company_num", "")
                if st.button("Generate audit report (PDF)", key=f"genrep_{_cnum}",
                             help="Build a one-page filed-evidence summary for this company."):
                    try:
                        _src = load_report_sources()
                        if _src is None:
                            st.caption("Report data not found "
                                       "(needs outputs/prospective_shap.csv and nlp/ files).")
                        else:
                            _shap_df, _llm_df, _spv_df, _entity_df = _src
                            build_company_pdf, _, report_filename = _report_api()
                            with st.spinner("Building report…"):
                                _pdf = build_company_pdf(
                                    _cnum, _shap_df, _llm_df, _spv_df, _entity_df,
                                    risk_score=None if np.isnan(risk) else float(risk))
                            _fname = report_filename(_cnum, _spv_df, _shap_df, _llm_df)
                            st.download_button(
                                "Download report (PDF)",
                                data=_pdf,
                                file_name=_fname,
                                mime="application/pdf",
                                key=f"dlrep_{_cnum}",
                            )
                            # No inline preview. Chrome blocks data: URLs inside
                            # an iframe, which is the only way to embed a PDF held
                            # in memory, so the preview renders as a blocked-page
                            # notice rather than the document. The download works.
                            st.caption(f"{len(_pdf) / 1024:.0f} KB, two pages: risk "
                                       f"rating, company profile, key risk indicators, "
                                       f"filing timeline, narrative summary, and "
                                       f"review guidance.")
                    except Exception as _e:
                        st.caption(f"Could not build the report for this company ({_e}).")


# ----------------------------------------------------------------------------
# TAB 4 - MODEL PERFORMANCE
# ----------------------------------------------------------------------------
if not IS_ENGAGEMENT:
    with tab4:
        st.markdown("#### Five-Model Comparison")
        st.caption("All models on identical train/test split. Test set never touched during HPO. "
                   "Optuna 100 trials × 5-fold CV per tree model.")

        if comp_df.empty:
            st.warning("model_comparison.csv not found.")
        else:
            if "avg_precision" in comp_df.columns:
                sorted_df = comp_df.sort_values("avg_precision", ascending=False).reset_index(drop=True)
            else:
                sorted_df = comp_df.reset_index(drop=True)

            disp_cols = [c for c in ("model", "avg_precision", "ap_ci_lo", "ap_ci_hi",
                                     "auc_roc", "auc_ci_lo", "auc_ci_hi",
                                     "f1", "ks_stat", "brier_score")
                         if c in sorted_df.columns]

            def highlight_first(row):
                if row.name == 0:
                    return [f"background-color: {UI_YELLOW}33; color: #FFF"] * len(row)
                return [""] * len(row)

            st.dataframe(
                sorted_df[disp_cols].style.apply(highlight_first, axis=1).format(precision=4),
                use_container_width=True, height=240,
            )

            if {"avg_precision", "auc_roc"}.issubset(sorted_df.columns):
                comp = sorted_df[["model", "avg_precision", "auc_roc"]].copy()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=comp["model"], x=comp["avg_precision"], orientation="h",
                    name="Average precision (rare-event detection)",
                    marker_color="#FFE600",
                    text=comp["avg_precision"].map("{:.3f}".format),
                    textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    y=comp["model"], x=comp["auc_roc"], orientation="h",
                    name="AUC-ROC (overall ranking)",
                    marker_color="#2E86DE",
                    text=comp["auc_roc"].map("{:.3f}".format),
                    textposition="outside",
                ))
                fig.update_layout(barmode="group", height=440,
                                  margin=dict(t=40, b=40, l=120, r=60),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font={"color": "#FFFFFF"},
                                  legend=dict(orientation="h", y=1.08, x=0),
                                  xaxis=dict(range=[0, 1.05], gridcolor="#2a2a3a"))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("XGBoost leads on average precision, the metric that matters for "
                           "catching rare dissolutions. Logistic Regression scores higher on "
                           "AUC but far lower on precision, which is why it is not selected.")

            st.markdown("---")
            st.markdown("##### Research Question Validation")

            rq_c1, rq_c2, rq_c3 = st.columns(3)
            with rq_c1:
                st.markdown(f"""<div class="rq-card">
<div class="h">RQ1 · AUC target ≥ 0.88</div>
<div class="v">{auc_str}</div>
<div class="s">PASSED · Du Jardin (2021) 0.80–0.88 exceeded</div>
</div>""", unsafe_allow_html=True)
        with rq_c2:
            st.markdown(f"""<div class="rq-card">
<div class="h">RQ2 · Median lead time</div>
<div class="v">14.6 months</div>
<div class="s">≥6mo flag 40.2% vs 5% baseline · p &lt; 1e-300</div>
</div>""", unsafe_allow_html=True)
        with rq_c3:
            st.markdown(f"""<div class="rq-card">
<div class="h">RQ3 · IF permutation</div>
<div class="v">p = 0.0000</div>
<div class="s">Observed AUC 0.5297 · null 95th 0.5080</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
<div style="background:{UI_DARK};padding:14px 18px;border-radius:8px;border-left:3px solid #1A73E8;">
<div style="color:#1A73E8;font-weight:600;font-size:0.85rem;margin-bottom:6px;">Class imbalance context</div>
<div style="color:#FFF;font-size:0.88rem;line-height:1.5;">
Train base rate 6.69% (98,926 rows) · test 4.07% (94,421 rows) · scale_pos_weight 13.94.
The CV-to-test AP gap (CV ~0.78 → test {ap_str}) is expected by design: train base rate is higher,
so test AP is harder. Both arms of every model see the same temporal split.
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Model card")
        st.caption("The governance factsheet a methodology or quality reviewer needs: "
                   "intended use, headline metrics, validation design, limitations and "
                   "data lineage. Built on demand from the locked evaluation results.")
        _card_py = ROOT / "build_model_card.py"
        if not _card_py.exists():
            st.caption(f"build_model_card.py not found next to {_THIS_FILE}.")
        elif st.button("Build model card (PDF)", key="gencard",
                       help="One-page methodology summary for Risk & Quality review."):
            try:
                from build_model_card import build_model_card
                with st.spinner("Building model card…"):
                    _card = build_model_card()
                st.download_button("Download model card (PDF)", data=_card,
                                   file_name="model_card.pdf",
                                   mime="application/pdf", key="dlcard")
                st.caption(f"{len(_card) / 1024:.0f} KB: intended use, headline "
                           f"metrics, validation design, limitations, and data "
                           f"lineage.")
            except Exception as _e:
                st.caption(f"Could not build the model card ({type(_e).__name__}: {_e}). "
                           "It needs reportlab: pip install reportlab")


# ----------------------------------------------------------------------------
# TAB 5 - SHAP INTELLIGENCE
# ----------------------------------------------------------------------------
if not IS_ENGAGEMENT:
    with tab5:
        st.markdown("#### Which Risk Factors Matter Most")
        st.caption("Ranked by how much each factor influences the model's dissolution-risk "
                   "assessment across the company population.")

        if mean_abs_shap is None or not feature_cols:
            st.warning("Risk-factor data not available. Run the explainability step first.")
        else:
            n_show = st.slider("Features to display", 10, len(feature_cols), 25)
            ranked = pd.DataFrame({
                "Feature": [feature_label(f) for f in feature_cols],
                "MeanAbsSHAP": mean_abs_shap,
            }).sort_values("MeanAbsSHAP", ascending=False).head(n_show)

            fig = px.bar(ranked.iloc[::-1], x="MeanAbsSHAP", y="Feature", orientation="h",
                         color_discrete_sequence=[UI_YELLOW])
            plotly_dark(fig, height=max(420, n_show * 18),
                        margin=dict(t=20, b=40, l=250, r=20),
                        xaxis_title="Overall influence on dissolution risk")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("##### Top 5 Risk Drivers")
            explanations = {
                "ar_filed_count": "Total Annual Returns filed across the company's life. Companies that stop filing are companies that are dying. Strongest single dissolution signal.",
                "company_age_years": "Years since incorporation. Both very young and very old extremes carry risk in different ways; the model learns the non-linear shape.",
                "annual_submission_rate": "Filings per year of tenure. Sustained low rates indicate operational drag; very high may indicate restructuring.",
                "total_submissions": "All-time submission count to CRO. Captures overall depth of engagement with the register.",
                "director_change_count": "Director appointments and resignations. Frequent changes correlate with governance instability and pre-dissolution churn.",
                "submission_history_years": "Length of CRO interaction history. Shorter histories carry less signal but elevated baseline risk.",
                "other_form_count": "Filings outside the standard AR / accounts cycle. Captures restructuring activity.",
                "age_vs_sector_median": "Company age relative to sector median. Sector-relative outliers carry signal.",
                "days_since_last_name_change": "Time since the last name change. Recent name changes correlate with restructuring or rebranding.",
                "name_change_count": "Total name changes. Frequent renaming correlates with corporate instability.",
            }
            import re as _re
            for _, r in ranked.head(5).iterrows():
                name = r["Feature"]; val = r["MeanAbsSHAP"]
                raw = _re.search(r"\(([^)]+)\)\s*$", name)
                raw_key = raw.group(1) if raw else name
                expl = explanations.get(raw_key, "Behavioural or structural factor derived from CRO filings.")
                st.markdown(f"""<div class="shap-driver">
<div style="display:flex;justify-content:space-between;align-items:center;">
<span style="color:{UI_YELLOW};font-weight:600;font-size:1.0rem;">{name}</span>
<span style="color:#FFF;font-family:monospace;">influence {val:.2f}</span>
</div>
<div style="color:{UI_TEXT_DIM};font-size:0.85rem;margin-top:6px;line-height:1.5;">{expl}</div>
</div>""", unsafe_allow_html=True)

    _conc = load_concordance_summary()
    if _conc:
        st.markdown("---")
        st.markdown("##### AI narrative faithfulness")
        st.caption("How closely the AI's per-company summaries track the model's own SHAP "
                   "drivers, where higher means the narrative emphasises what actually moves the score.")
        k1, k2, k3 = st.columns(3)
        if "precision_top5" in _conc:
            k1.metric("Cited features that are top-5 drivers",
                      f"{_conc['precision_top5'] * 100:.1f}%")
        if "cites_any_top3" in _conc:
            k2.metric("Narratives citing a top-3 driver",
                      f"{_conc['cites_any_top3'] * 100:.1f}%")
        if "leads_with_top_driver" in _conc:
            k3.metric("Lead with the top driver",
                      f"{_conc['leads_with_top_driver'] * 100:.1f}%")


# ----------------------------------------------------------------------------
# TAB 6 - STAGE 2 NLP
# ----------------------------------------------------------------------------
if IS_ENGAGEMENT:
    with tab7:
        with safe_block("Client Portfolio"):
            render_portfolio_tab()


if not IS_ENGAGEMENT:
    with tab6:
        with safe_block("Model Validation"):
            render_stage2_tab()


# Footer
st.markdown(f"""
<div style="text-align:center; color:{UI_TEXT_DIM}; font-size:0.75rem;
            margin-top:30px; padding:18px; border-top:1px solid {UI_MID};">
Irish SME Dissolution Risk Dashboard
</div>
""", unsafe_allow_html=True)
