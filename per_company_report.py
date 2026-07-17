"""
Per-company audit report generator (six-section PDF).

Builds a filed-evidence risk summary for a single company, structured as:
  1. Executive summary and overall risk rating
  2. Company profile
  3. Key risk indicators (SHAP contribution chart)
  4. Filing-behaviour trends
  5. Narrative observations (language-model reasoning trace)
  6. Audit considerations and areas for further review

PDF is used so the summary is a fixed, human-reviewable record suitable for filing
as workpaper evidence. Returns the document as bytes so it can be served directly
from a download button, or writes a file when run from the command line.

Reads:
  outputs/prospective_shap.csv               tier, score, base value, top_drivers_json
  outputs/nlp/llm_features.csv               narrative, distress signals, audit steps
  outputs/nlp/prospective_spv_labelled.csv   profile, filing features, SPV + silence flags
  outputs/nlp/entity_types.csv               entity-type classification
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, KeepTogether)

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
NLP_DIR = OUTPUTS_DIR / "nlp"
CAND_DIRS = [OUTPUTS_DIR, NLP_DIR, ROOT]

# Cohort lead-time result (RQ2): comparable dissolved companies were identifiable a
# median of this many months before dissolution.
RQ2_MEDIAN_LEAD_MONTHS = 14.6

CHARCOAL = colors.HexColor("#2E2E38")
GREY = colors.HexColor("#6B6B76")
LIGHT = colors.HexColor("#F2F2F4")
WHITE = colors.white

TIER_COLORS = {
    "PRIORITY": colors.HexColor("#C1121F"),
    "DISSOLUTION_RISK": colors.HexColor("#E07B00"),
    "BEHAVIORAL_ANOMALY": colors.HexColor("#1A73E8"),
    "LOW_CONCERN": colors.HexColor("#1A7340"),
}
TIER_LABEL = {
    "PRIORITY": "Priority", "DISSOLUTION_RISK": "Dissolution risk",
    "BEHAVIORAL_ANOMALY": "Behavioural anomaly", "LOW_CONCERN": "Low concern",
}
ELEVATED = {"PRIORITY", "DISSOLUTION_RISK"}

FILING_FIELDS = [
    ("ar_filed_count", "Annual returns filed", 0),
    ("total_submissions", "Total submissions", 0),
    ("annual_submission_rate", "Annual submission rate", 2),
    ("submission_history_years", "Filing history (years)", 1),
    ("director_change_count", "Director changes", 0),
    ("days_since_last_ar_filing", "Days since last AR", 0),
]

# NACE Rev. 2 section names keyed by division range (first two digits of the code),
# used to name a sector when the source label is missing or "Unknown".
NACE_SECTIONS = [
    (1, 3, "Agriculture, forestry and fishing"),
    (5, 9, "Mining and quarrying"),
    (10, 33, "Manufacturing"),
    (35, 35, "Electricity, gas, steam and air conditioning supply"),
    (36, 39, "Water supply, sewerage and waste management"),
    (41, 43, "Construction"),
    (45, 47, "Wholesale and retail trade"),
    (49, 53, "Transportation and storage"),
    (55, 56, "Accommodation and food service activities"),
    (58, 63, "Information and communication"),
    (64, 66, "Financial and insurance activities"),
    (68, 68, "Real estate activities"),
    (69, 75, "Professional, scientific and technical activities"),
    (77, 82, "Administrative and support service activities"),
    (84, 84, "Public administration and defence"),
    (85, 85, "Education"),
    (86, 88, "Human health and social work activities"),
    (90, 93, "Arts, entertainment and recreation"),
    (94, 96, "Other service activities"),
    (97, 98, "Activities of households as employers"),
    (99, 99, "Activities of extraterritorial organisations"),
]


def nace_sector_name(code):
    try:
        c = int(float(code))
    except (ValueError, TypeError):
        return None
    div = int(str(c).zfill(2)[:2])
    for lo, hi, name in NACE_SECTIONS:
        if lo <= div <= hi:
            return name
    return None


def _key(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip().lstrip("0") or "0"


def _num(v, nd=0):
    try:
        f = float(v)
        return f"{f:,.{nd}f}" if nd else f"{int(round(f)):,}"
    except (ValueError, TypeError):
        return "n/a"


def _fmt_val(v):
    try:
        return f"{float(v):g}"
    except (ValueError, TypeError):
        return str(v)


def _truthy(v):
    return str(v).strip() in ("1", "1.0", "True", "true", "yes", "Yes")


def _pretty(feat):
    return feat.replace("_", " ").strip()


def _lookup(df, company_num):
    if df is None or df.empty:
        return {}
    hit = df[df["_k"] == _key(company_num)]
    return hit.iloc[0].to_dict() if len(hit) else {}


# --------------------------------------------------------------------------- #
# SHAP contribution chart
# --------------------------------------------------------------------------- #
def _waterfall_png(base, drivers, shap_sum, top_n=8):
    """Render a diverging SHAP contribution chart (log-odds) to PNG bytes.

    Each bar starts at zero: red extends right (increases risk), green extends
    left (reduces risk), sorted by impact. This reads more clearly for a
    non-technical reviewer than a cumulative waterfall while showing the same
    SHAP values. base and shap_sum are accepted for signature stability.
    """
    drivers = sorted(drivers, key=lambda t: abs(t[1]), reverse=True)[:top_n]
    labels = [_pretty(f) for f, sh, val in drivers][::-1]
    vals = [sh for _, sh, _ in drivers][::-1]
    n = len(vals)

    fig, ax = plt.subplots(figsize=(6.6, 0.44 * n + 0.9), dpi=150)
    span = max((abs(v) for v in vals), default=1) or 1
    for i, sh in enumerate(vals):
        col = "#C1121F" if sh > 0 else "#1A7340"
        ax.barh(i, sh, color=col, height=0.62, zorder=3)
        off = 0.05 * span
        ax.text(sh + (off if sh >= 0 else -off), i, f"{sh:+.2f}", va="center",
                ha="left" if sh >= 0 else "right", fontsize=8, color="#2E2E38")

    ax.axvline(0.0, color="#7A7A85", lw=1.1, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5, color="#2E2E38")
    ax.set_xlabel("SHAP contribution to risk (log-odds)", fontsize=8.5, color="#6B6B76")
    ax.tick_params(axis="x", labelsize=8, colors="#6B6B76")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D8D8DE")
    ax.margins(x=0.20, y=0.06)
    fig.tight_layout(pad=0.5)
    out = io.BytesIO()
    fig.savefig(out, format="png", bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def build_company_pdf(company_num, shap_df, llm_df, spv_df, entity_df, risk_score=None):
    d = {}
    d.update(_lookup(spv_df, company_num))
    d.update({k: v for k, v in _lookup(shap_df, company_num).items() if pd.notna(v)})
    d.update({k: v for k, v in _lookup(llm_df, company_num).items() if pd.notna(v)})
    d.setdefault("company_num", company_num)
    entity = _lookup(entity_df, company_num)

    tier = str(d.get("combined_risk_tier", ""))
    tcol = TIER_COLORS.get(tier, GREY)
    # The Stage 1 isotonic-calibrated probability, which is what the dashboard
    # gauge shows and what every result in the write-up is computed on. An
    # explicitly passed value wins so the caller and the report cannot diverge.
    _score_val = risk_score if risk_score is not None else d.get("dissolution_risk_score")
    try:
        score_txt = f"{float(_score_val) * 100:.1f}"
    except (ValueError, TypeError):
        score_txt = "n/a"
    spv = _truthy(d.get("is_likely_spv"))
    silence = _truthy(d.get("silent_yr1")) or (str(d.get("filed_current_year")) in ("0", "0.0"))

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13, textColor=CHARCOAL)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=GREY)
    h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=11,
                       textColor=CHARCOAL, spaceBefore=10, spaceAfter=4)
    kicker = ParagraphStyle("kicker", parent=small, fontName="Helvetica-Bold")
    name_st = ParagraphStyle("name", parent=body, fontName="Helvetica-Bold",
                             fontSize=17, leading=19)
    badge_lbl = ParagraphStyle("bl", parent=body, fontName="Helvetica-Bold",
                               fontSize=7.5, textColor=WHITE)
    badge_tier = ParagraphStyle("bt", parent=body, fontName="Helvetica-Bold",
                                fontSize=15, textColor=WHITE, leading=17)
    badge_sc = ParagraphStyle("bs", parent=body, fontSize=8.5, textColor=WHITE)

    story = []

    # Header: identity (left) + risk badge (right)
    ident = [Paragraph("Irish SME Dissolution Risk&nbsp;&nbsp;|&nbsp;&nbsp;Confidential audit summary", kicker),
             Spacer(1, 3),
             Paragraph(str(d.get("company_name", "")), name_st),
             Spacer(1, 2),
             Paragraph(f"CRO number {d.get('company_num', '')}", small)]
    badge = Table([[Paragraph("OVERALL RISK RATING", badge_lbl)],
                   [Paragraph(TIER_LABEL.get(tier, tier or "n/a"), badge_tier)],
                   [Paragraph(f"Dissolution risk: {score_txt}%", badge_sc)]],
                  colWidths=[52 * mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tcol),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 1),
    ]))
    header = Table([[ident, badge]], colWidths=[112 * mm, 56 * mm])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [header, Spacer(1, 8)]

    # Section 1: Executive summary
    story.append(Paragraph("1&nbsp;&nbsp;Executive summary", h))
    summ = (f"This company is ranked {TIER_LABEL.get(tier, tier).lower()}, with a "
            f"calibrated dissolution probability of {score_txt}% within twenty-four "
            f"months of its observation date, against a base rate of 4.07% across the "
            f"modelled population. The assessment rests on the company&#8217;s "
            f"filing-behaviour profile alone; no financial statement content is used.")
    try:
        _sv = float(_score_val)
    except (ValueError, TypeError):
        _sv = None
    if _sv is not None and _sv >= 0.9995:
        summ += (" This figure is the upper bound of the calibrated scale rather than "
                 "a statement of certainty: the calibration is bounded by the "
                 "dissolution rate observed in each score band, and every company in "
                 "the highest band dissolved.")
    elif _sv is not None and _sv <= 0.0005:
        summ += (" This figure is the lower bound of the calibrated scale rather than "
                 "a statement that the company will not dissolve: no company in the "
                 "lowest score band dissolved, so the calibration returns its floor. "
                 "Absence of signal is not clearance.")
    story.append(Paragraph(summ, body))

    # Section 2: Company profile
    story.append(Paragraph("2&nbsp;&nbsp;Company profile", h))
    label = str(d.get("nace_section_label") or "").strip()
    if label.lower() in ("", "unknown", "un", "nan", "none"):
        label = nace_sector_name(d.get("nace_v2_code")) or "Unknown"
    nace_code = _num(d.get("nace_v2_code"))
    sector_disp = f"{label} ({nace_code})" if nace_code != "n/a" else label
    prof = [
        ("Registered name", str(d.get("company_name", "n/a"))),
        ("CRO number", str(d.get("company_num", "n/a"))),
        ("Status", str(d.get("company_status", "n/a")).strip()),
        ("Company type", str(d.get("company_type", "n/a"))),
        ("Sector (NACE)", sector_disp),
        ("County", str(d.get("county", "n/a"))),
        ("Company age (years)", _num(d.get("company_age_years"), 1)),
        ("Directors on record", _num(d.get("director_count"))),
        ("Entity type (classifier)", str(entity.get("entity_type", "n/a")).replace("_", " ")),
        ("Likely special-purpose vehicle", "Yes" if spv else "No"),
    ]
    prof_rows = [[Paragraph(k, small), Paragraph(f"<b>{v}</b>", body)] for k, v in prof]
    pt = Table(prof_rows, colWidths=[55 * mm, 113 * mm])
    pt.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, WHITE]),
    ]))
    story.append(pt)

    # Section 3: Key risk indicators (SHAP contribution chart)
    story.append(Paragraph("3&nbsp;&nbsp;Key risk indicators", h))
    drivers = []
    try:
        for x in json.loads(d.get("top_drivers_json", "[]")):
            drivers.append((x.get("feature", ""), float(x.get("shap", 0)), x.get("value", "")))
    except (TypeError, ValueError):
        drivers = []
    if drivers:
        base = float(d.get("base_value", 0) or 0)
        shap_sum = float(d.get("shap_sum", sum(s for _, s, _ in drivers)) or 0)
        img = Image(_waterfall_png(base, drivers, shap_sum))
        img._restrictSize(165 * mm, 95 * mm)
        story.append(KeepTogether([img]))
        story.append(Paragraph(
            "Each bar is a feature's contribution to the model output in log-odds; "
            "<font color='#C1121F'>red increases</font> and "
            "<font color='#1A7340'>green reduces</font> dissolution risk. These "
            "contributions explain the ranking and do not sum to the risk score above.", small))
    else:
        story.append(Paragraph("No driver breakdown available for this company.", body))

    # Section 4: Filing-behaviour trends
    cells = []
    for field, label, nd in FILING_FIELDS:
        cells.append((label, _num(d.get(field), nd)))
    cells.append(("Late-filer flag", "Yes" if _truthy(d.get("late_filer_flag")) else "No"))
    cells.append(("Name changes", _num(d.get("name_change_count"))))
    val_st = ParagraphStyle('v', parent=body, fontName="Helvetica-Bold", fontSize=13)
    grid = []
    for i in range(0, len(cells), 4):
        chunk = cells[i:i + 4]
        vals = [Paragraph(v, val_st) for _, v in chunk]
        labs = [Paragraph(l, small) for l, _ in chunk]
        grid.append(vals + [""] * (4 - len(vals)))
        grid.append(labs + [""] * (4 - len(labs)))
    ft = Table(grid, colWidths=[42 * mm] * 4)
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # keep each value directly above its label with no inter-row gap
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
    ]))
    story.append(KeepTogether([Paragraph("4&nbsp;&nbsp;Filing-behaviour trends", h), ft]))

    # Section 5: Narrative observations
    story.append(Paragraph("5&nbsp;&nbsp;Narrative observations", h))
    story.append(Paragraph(str(d.get("audit_narrative", "") or "No narrative available."), body))
    if silence:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<i>Note: 2024 filing data is partial at the time of analysis. Any recent "
            "filing-silence signal is a probabilistic indicator, not a confirmed "
            "non-filing event, and should be verified against the live CRO record.</i>", small))

    # Section 6: Audit considerations
    story.append(Paragraph("6&nbsp;&nbsp;Audit considerations and further review", h))
    steps = str(d.get("audit_steps", "") or "").strip()
    step_items = [x.strip() for x in steps.replace(";", "\n").splitlines() if x.strip()]
    left = []
    if step_items:
        for i, st_ in enumerate(step_items[:6], 1):
            left.append(Paragraph(f"{i}.&nbsp;&nbsp;{st_}", body))
            left.append(Spacer(1, 2))
    else:
        left.append(Paragraph("No specific steps recorded.", body))
    if spv:
        note = ("Rule-based screening flags this entity as a likely special-purpose "
                "vehicle (financing / securitisation / leasing DAC). For these, a "
                "scheduled dissolution at deal-end is normal and does not indicate "
                "distress. Confirm the entity type before escalation.")
    else:
        note = ("No special-purpose-vehicle signal on this entity. Treat an elevated "
                "rating as a genuine dissolution-risk indicator and corroborate against "
                "current CRO status and recent filings.")
    conf = str(d.get("confidence", "") or "n/a")
    right = [Paragraph("<b>Review guidance</b>", body), Spacer(1, 3),
             Paragraph(note, small), Spacer(1, 6),
             Paragraph(f"<b>Narrative confidence:</b> {conf}", small)]
    if tier in ELEVATED:
        right += [Spacer(1, 4),
                  Paragraph(f"<b>Estimated lead time:</b> ~{RQ2_MEDIAN_LEAD_MONTHS} "
                            f"months", small)]
    audit = Table([[left, Table([[r] for r in right], colWidths=[54 * mm])]],
                  colWidths=[108 * mm, 60 * mm])
    audit.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (1, 0), (1, 0), LIGHT),
        ("LEFTPADDING", (1, 0), (1, 0), 8), ("RIGHTPADDING", (1, 0), (1, 0), 8),
        ("TOPPADDING", (1, 0), (1, 0), 8), ("BOTTOMPADDING", (1, 0), (1, 0), 8),
    ]))
    story.append(audit)
    story.append(Spacer(1, 8))
    story.append(Paragraph("For internal audit triage only.", small))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=12 * mm,
                            title=f"Risk summary {company_num}")
    doc.build(story)
    return buf.getvalue()


def _read(path):
    if path is None:
        return None
    if not Path(path).exists():
        print(f"WARNING: data file not found: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path, low_memory=False)
    df["_k"] = df["company_num"].apply(_key)
    return df


def _find(filename):
    for dr in CAND_DIRS:
        p = dr / filename
        if p.exists():
            return p
    return None


def report_filename(company_num, *dfs, ext="pdf"):
    """Return a 'CompanyName-ID.ext' filename, sanitised for Windows/macOS."""
    name = ""
    for df in dfs:
        r = _lookup(df, company_num)
        if r.get("company_name"):
            name = str(r["company_name"])
            break
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "company"
    return f"{safe}-{_key(company_num)}.{ext}"


def load_sources(shap=None, llm=None, spv=None, entity=None):
    return (_read(shap or _find("prospective_shap.csv")),
            _read(llm or _find("llm_features.csv")),
            _read(spv or _find("prospective_spv_labelled.csv")),
            _read(entity or _find("entity_types.csv")))


def main():
    ap = argparse.ArgumentParser(description="Per-company audit report (PDF)")
    ap.add_argument("company_num")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shap", default=None)
    ap.add_argument("--llm", default=None)
    ap.add_argument("--spv", default=None)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    shap_df, llm_df, spv_df, entity_df = load_sources(
        args.shap, args.llm, args.spv, args.entity)
    data = build_company_pdf(args.company_num, shap_df, llm_df, spv_df, entity_df)
    out = args.out or report_filename(args.company_num, spv_df, shap_df, llm_df)
    Path(out).write_bytes(data)
    print(f"Wrote {out} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
