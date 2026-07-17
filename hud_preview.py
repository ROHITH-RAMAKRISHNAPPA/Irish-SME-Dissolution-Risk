"""
Styling preview for the Irish SME Dissolution Risk Dashboard.

Standalone. Reads nothing, writes nothing, imports nothing from app.py. Every
figure shown is a real V17 result, hardcoded here only so the preview can run
without the pipeline.

    streamlit run hud_preview.py

The point is to look at the treatment side by side with the live dashboard and
decide whether it belongs in an EY meeting room.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

st.set_page_config(page_title="Styling preview", page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

# EY brand palette, as already used by the dashboard.
EY_YELLOW = "#FFE600"
EY_CHARCOAL = "#2E2E38"
EY_BLACK = "#1A1A23"
EY_MID = "#3D3D4E"
EY_DIM = "#A0A0B0"

GREEN, AMBER, BLUE, RED = "#1A7340", "#E07B00", "#1A73E8", "#C1121F"

# Real V17 results.
AP, AUC = 0.6298, 0.9412
COHORT = 28974
TIERS = [("PRIORITY", 78, RED), ("DISSOLUTION_RISK", 5833, AMBER),
         ("BEHAVIOURAL_ANOMALY", 1578, BLUE), ("LOW_CONCERN", 21485, GREEN)]

_FIRST = "ui_hud_painted" not in st.session_state
st.session_state["ui_hud_painted"] = True
ENTER = "enter" if _FIRST else ""

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
  --mono:"IBM Plex Mono",Consolas,monospace;
  --sans:"IBM Plex Sans","Segoe UI",sans-serif;
  --ease:cubic-bezier(.2,.7,.3,1);
}}

/* A faint measured grid behind everything, plus a single warm bloom off the
   top-left. Both sit far enough back to read as texture rather than pattern. */
.stApp {{
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(255,230,0,.10), transparent 62%),
    radial-gradient(900px 480px at 96% 4%, rgba(61,61,78,.55), transparent 60%),
    linear-gradient(180deg, {EY_BLACK} 0%, #15151C 100%);
  background-attachment: fixed;
}}
.stApp::before {{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(circle at 50% 34%, #000 12%, transparent 76%);
}}
.block-container {{ max-width:1500px; padding-top:1.2rem; position:relative; z-index:1; }}
h1,h2,h3,h4,h5 {{ font-family:var(--sans); color:#FFF; }}

@keyframes rise {{ from{{opacity:0;transform:translateY(10px);}} to{{opacity:1;transform:none;}} }}
@keyframes breathe {{ 0%,100%{{opacity:.3;transform:scale(1);}} 50%{{opacity:1;transform:scale(1.4);}} }}
@keyframes sweep {{ from{{transform:translateX(-110%);}} to{{transform:translateX(310%);}} }}
.enter .g {{ animation: rise .5s var(--ease) both; }}
.enter .g:nth-child(2){{animation-delay:.06s}} .enter .g:nth-child(3){{animation-delay:.12s}}
.enter .g:nth-child(4){{animation-delay:.18s}}
@media (prefers-reduced-motion:reduce){{ .enter .g,.glass::after{{animation:none!important}} }}

/* Glass: a translucent plate over the gradient, a hairline top edge where the
   light would catch, and a slow specular sweep on the hero only. */
.glass {{
  position:relative; overflow:hidden;
  background: linear-gradient(155deg, rgba(255,255,255,.055), rgba(255,255,255,.012));
  backdrop-filter: blur(13px) saturate(115%);
  -webkit-backdrop-filter: blur(13px) saturate(115%);
  border:1px solid rgba(255,255,255,.09);
  border-radius:12px;
  box-shadow: 0 10px 34px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.10);
  transition: transform .2s var(--ease), box-shadow .2s var(--ease),
              border-color .2s var(--ease);
}}
.glass:hover {{
  transform: translateY(-3px);
  border-color: rgba(255,230,0,.30);
  box-shadow: 0 16px 44px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.16),
              0 0 0 1px rgba(255,230,0,.10);
}}
.hero::after {{
  content:""; position:absolute; top:0; left:0; width:26%; height:100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.05), transparent);
  animation: sweep 9s ease-in-out infinite; pointer-events:none;
}}

.kpi {{ padding:17px 19px; border-left:3px solid {EY_YELLOW}; }}
.kpi .lab {{ font-size:.63rem; text-transform:uppercase; letter-spacing:.11em;
             color:{EY_DIM}; font-weight:600; }}
.kpi .val {{ font-family:var(--mono); font-variant-numeric:tabular-nums;
             font-size:1.95rem; color:#FFF; margin:7px 0 3px; letter-spacing:-.02em; }}
.kpi .sub {{ font-size:.72rem; color:{EY_DIM}; font-family:var(--mono); }}

.pulse {{ display:inline-block;width:6px;height:6px;border-radius:50%;
          margin-right:8px;vertical-align:middle;animation:breathe 2.6s ease-in-out infinite; }}
.tag {{ font-family:var(--mono); font-size:.66rem; letter-spacing:.09em;
        text-transform:uppercase; color:{EY_DIM}; }}
hr {{ border-color: rgba(255,255,255,.07); }}
"""

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
    "<style>" + _css_min(_CSS) + "</style>",
    unsafe_allow_html=True,
)


def countup(value, suffix="", decimals=0, label="", sub="", accent=EY_YELLOW,
            saturated=False, height=124):
    """A KPI that counts to its value.

    A saturated score is shown outright rather than counted to: animating a
    number towards 100% dramatises a calibration ceiling into a certainty the
    model does not claim.
    """
    if saturated:
        js_val, start = f'"{value}{suffix}"', "null"
    else:
        js_val, start = "null", value
    uid = abs(hash((label, value, sub))) % 100000
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:transparent; font-family:"IBM Plex Sans",sans-serif; }}
  .c {{ position:relative; overflow:hidden; height:{height - 8}px; box-sizing:border-box;
        padding:16px 19px; border-radius:12px; border-left:3px solid {accent};
        background: linear-gradient(155deg, rgba(255,255,255,.055), rgba(255,255,255,.012));
        backdrop-filter: blur(13px); border-top:1px solid rgba(255,255,255,.09);
        border-right:1px solid rgba(255,255,255,.05);
        border-bottom:1px solid rgba(255,255,255,.05);
        box-shadow: 0 10px 30px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.10);
        transition: transform .2s cubic-bezier(.2,.7,.3,1), box-shadow .2s, border-color .2s; }}
  .c:hover {{ transform:translateY(-3px); border-color:rgba(255,230,0,.3);
              box-shadow:0 16px 40px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.16); }}
  .lab {{ font-size:.63rem; text-transform:uppercase; letter-spacing:.11em;
          color:{EY_DIM}; font-weight:600; }}
  .val {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
          font-size:1.95rem; color:#FFF; margin:8px 0 3px; letter-spacing:-.02em; }}
  .sub {{ font-size:.72rem; color:{EY_DIM}; font-family:"IBM Plex Mono",monospace; }}
</style>
<div class="c">
  <div class="lab">{label}</div>
  <div class="val" id="v{uid}">0</div>
  <div class="sub">{sub}</div>
</div>
<script>
(function(){{
  var el = document.getElementById("v{uid}");
  var fixed = {js_val};
  if (fixed !== null) {{ el.textContent = fixed; return; }}
  var target = {start}, dp = {decimals}, dur = 900, t0 = null;
  function fmt(x){{ return x.toLocaleString(undefined,
      {{minimumFractionDigits:dp, maximumFractionDigits:dp}}) + "{suffix}"; }}
  function step(ts){{
    if(!t0) t0 = ts;
    var p = Math.min((ts - t0) / dur, 1);
    var e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(target * e);
    if(p < 1) requestAnimationFrame(step); else el.textContent = fmt(target);
  }}
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) el.textContent = fmt(target);
  else requestAnimationFrame(step);
}})();
</script>""", height=height)


st.markdown(f"""
<div class="{ENTER}"><div class="g glass hero" style="padding:22px 28px;margin-bottom:16px;
     border-left:4px solid {EY_YELLOW};">
  <h1 style="margin:0;font-size:1.5rem;letter-spacing:-.5px;">Irish SME Dissolution Risk</h1>
  <div class="tag" style="margin-top:8px;">
    <span class="pulse" style="background:{GREEN};box-shadow:0 0 8px {GREEN};"></span>
    All artefacts loaded
    <span style="opacity:.35;margin:0 8px;">|</span>Scored at the 31 December 2024 observation date
    <span style="opacity:.35;margin:0 8px;">|</span>XGBoost AP {AP:.4f} / AUC {AUC:.4f}
    <span style="opacity:.35;margin:0 8px;">|</span>{COHORT:,} companies, 84 features
  </div>
</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="tag">Counting KPIs &mdash; watch them arrive</div>',
            unsafe_allow_html=True)
c = st.columns(4)
with c[0]:
    countup(COHORT, label="Companies scored", sub="prospective cohort")
with c[1]:
    countup(78, label="Priority", sub="both stages agree", accent=RED)
with c[2]:
    countup(AP * 100, suffix="%", decimals=2, label="Average precision",
            sub=f"AUC {AUC:.4f}", accent=EY_YELLOW)
with c[3]:
    countup(14.6, decimals=1, label="Median lead time", sub="months before dissolution",
            accent=GREEN)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="tag">A saturated score is shown, never counted to</div>',
            unsafe_allow_html=True)
c = st.columns(4)
with c[0]:
    countup(17.3, suffix="%", decimals=1, label="Dissolution risk", sub="Top 83.3% of cohort")
with c[1]:
    countup("100.0", suffix="%", label="Dissolution risk", saturated=True,
            sub="ceiling of the scale, not certainty", accent=RED)
with c[2]:
    countup("<0.1", suffix="%", label="Dissolution risk", saturated=True,
            sub="floor: absence is not clearance", accent=GREEN)
with c[3]:
    countup(0.93, decimals=2, label="Anomaly score", sub="relative, not a probability",
            accent=BLUE)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([3, 2])
with left:
    st.markdown('<div class="tag">Glass panel over the gradient</div>',
                unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        y=[t[0] for t in TIERS][::-1], x=[t[1] for t in TIERS][::-1],
        orientation="h", marker_color=[t[2] for t in TIERS][::-1],
        text=[f"{t[1]:,}" for t in TIERS][::-1], textposition="outside"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300, showlegend=False,
        margin=dict(t=18, b=34, l=170, r=70),
        font=dict(family="IBM Plex Sans", color="#FFF"),
        xaxis=dict(type="log", title="Companies (log scale)",
                   tickfont=dict(family="IBM Plex Mono", size=11),
                   gridcolor="rgba(255,255,255,.06)"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11)),
        hoverlabel=dict(bgcolor=EY_CHARCOAL, bordercolor=EY_YELLOW,
                        font=dict(family="IBM Plex Mono", color="#FFF")),
        transition=dict(duration=320, easing="cubic-in-out"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="tag">Hover a card</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="glass" style="padding:16px 19px;margin-bottom:11px;border-left:3px solid {EY_YELLOW};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="color:{EY_YELLOW};font-weight:600;">ar_filed_count</span>
    <span style="font-family:var(--mono);color:#FFF;">influence 3.73</span>
  </div>
  <div style="color:{EY_DIM};font-size:.82rem;margin-top:6px;line-height:1.55;">
    Annual returns filed. The strongest single indicator in the model.</div>
</div>
<div class="glass" style="padding:16px 19px;margin-bottom:11px;border-left:3px solid {EY_YELLOW};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="color:{EY_YELLOW};font-weight:600;">company_age_years</span>
    <span style="font-family:var(--mono);color:#FFF;">influence 3.11</span>
  </div>
  <div style="color:{EY_DIM};font-size:.82rem;margin-top:6px;line-height:1.55;">
    Younger companies carry elevated baseline risk.</div>
</div>
<div class="glass" style="padding:16px 19px;border-left:3px solid {EY_YELLOW};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="color:{EY_YELLOW};font-weight:600;">annual_submission_rate</span>
    <span style="font-family:var(--mono);color:#FFF;">influence 2.67</span>
  </div>
  <div style="color:{EY_DIM};font-size:.82rem;margin-top:6px;line-height:1.55;">
    Filing frequency. Sustained decline precedes dissolution.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Preview only. Every figure shown is a real V17 result. Reload the page to "
           "see the entrance sequence again; it fires once per session, not on every "
           "interaction.")
