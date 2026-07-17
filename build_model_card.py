"""
Model card generator for the Irish SME Dissolution Risk model.

Produces a static one/two-page PDF factsheet for methodology and quality
reviewers: intended use, headline metrics, validation design, limitations and
data lineage. This is the model-level governance artifact (as distinct from the
per-company audit report, which describes a single company).

Run:
    python build_model_card.py                 -> writes model_card.pdf
    python build_model_card.py --out card.pdf

The headline figures below are the model's locked evaluation results. Edit the
MODEL_FACTS block if the underlying evaluation is re-run and the numbers move.
"""

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

CHARCOAL = "#2E2E38"
GREY = "#6B6B76"
LIGHT = "#F2F2F4"
ACCENT = "#1A7340"

# --------------------------------------------------------------------------- #
# Model facts (locked evaluation results)
# --------------------------------------------------------------------------- #
MODEL_FACTS = {
    "name": "Irish SME Dissolution Risk model",
    "purpose": (
        "Ranks active Irish SMEs by their likelihood of statutory dissolution "
        "using public Companies Registration Office (CRO) filing-behaviour "
        "metadata only, to support early, evidence-based audit triage."),
    "algorithm": "Gradient-boosted decision trees (XGBoost), selected over "
                 "LightGBM, logistic regression, random forest and decision "
                 "tree baselines on average precision.",
    "population": "814,836 companies in the Irish register; 28,974 active "
                  "companies scored prospectively.",
    "splits": "98,926 train / 94,421 test / 28,974 prospective.",
    "base_rates": "Dissolution base rate 6.69% (train) and 4.07% (test).",
    "class_weight": "scale_pos_weight = 13.94 to correct class imbalance.",
    # Headline metrics
    "ap": "0.6298",
    "auc": "0.9412",
    "parsimony": "Top-15 features retain 89.3% of average precision.",
    # Calibration / anomaly
    "calibration": "Isotonic calibration applied to map scores to probabilities.",
    # Research findings
    "rq2": "For companies that later dissolved, the risk was identifiable a "
           "median of 14.6 months before statutory dissolution; 75.7% of "
           "six-month-lead dissolutions were flagged (p < 0.000001).",
    "rq3": "Isolation Forest anomaly layer AUC 0.5297 vs a null of 0.5080 "
           "(permutation test, p < 0.0001).",
    "ablation": "Dropping all financial features moves average precision only "
                "marginally (0.633 to 0.611), confirming that filing-behaviour "
                "metadata alone carries the predictive signal.",
    "top_features": ("ar_filed_count, company_age_years, annual_submission_rate, "
                     "total_submissions, director_change_count"),
    "leakage": "Winsorisation and all feature scaling are fit on the training "
               "fold only; no test or prospective information enters training. "
               "Temporal separation is enforced between the observation window "
               "and the dissolution outcome.",
    "limitations": [
        "The model ranks relative dissolution risk; it does not assert that a "
        "high-scoring company will dissolve, nor does it establish cause.",
        "Rule-based screening shows many top-ranked companies are special-purpose "
        "vehicles, whose dissolution is often a scheduled deal-end rather than "
        "distress; these require entity-type confirmation before escalation.",
        "Coverage of financial statements is partial across the register, so the "
        "model is deliberately built on filing-behaviour metadata rather than "
        "financials.",
        "The most recent filing year can be partial at scoring time, so recent "
        "filing-silence signals are probabilistic and should be verified against "
        "the live CRO record.",
    ],
    "lineage": [
        ("Primary data", "CRO public filing-behaviour metadata (annual returns, "
                         "submissions, director changes, company age and status)."),
        ("Supplementary", "Orbis financials where available (used only for the "
                          "ablation study, not in the deployed model)."),
        ("Outcome label", "Statutory dissolution status from the CRO register."),
        ("Splits", "Train / test / prospective as above, with training-fold-only "
                   "preprocessing."),
    ],
}


def build_model_card(facts=MODEL_FACTS):
    from io import BytesIO
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Model card")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13, textColor=CHARCOAL)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=GREY)
    h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=11,
                       textColor=CHARCOAL, spaceBefore=11, spaceAfter=4)
    title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold",
                           fontSize=17, leading=20)
    kicker = ParagraphStyle("kicker", parent=small, fontName="Helvetica-Bold")

    def kv_table(rows):
        t = Table([[Paragraph(k, small), Paragraph(v, body)] for k, v in rows],
                  colWidths=[42 * mm, 126 * mm])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, "#E2E2E6"),
        ]))
        return t

    s = []
    s.append(Paragraph("Model card&nbsp;&nbsp;|&nbsp;&nbsp;Confidential methodology summary", kicker))
    s.append(Spacer(1, 3))
    s.append(Paragraph(facts["name"], title))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Intended use", h))
    s.append(Paragraph(facts["purpose"], body))

    s.append(Paragraph("Model and data", h))
    s.append(kv_table([
        ("Algorithm", facts["algorithm"]),
        ("Population", facts["population"]),
        ("Data splits", facts["splits"]),
        ("Base rates", facts["base_rates"]),
        ("Class balancing", facts["class_weight"]),
        ("Top features", facts["top_features"]),
    ]))

    s.append(Paragraph("Performance", h))
    perf = Table([
        [Paragraph("Average precision", small), Paragraph(facts["ap"], body),
         Paragraph("ROC AUC", small), Paragraph(facts["auc"], body)],
    ], colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm])
    perf.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                              ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                              ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    s.append(perf)
    s.append(Spacer(1, 3))
    s.append(Paragraph("&bull;&nbsp;&nbsp;" + facts["parsimony"], body))
    s.append(Paragraph("&bull;&nbsp;&nbsp;" + facts["calibration"], body))
    s.append(Paragraph("&bull;&nbsp;&nbsp;" + facts["ablation"], body))

    s.append(Paragraph("Validation design", h))
    s.append(Paragraph(facts["leakage"], body))
    s.append(Spacer(1, 3))
    s.append(Paragraph("<b>Lead-time validation.</b> " + facts["rq2"], body))
    s.append(Spacer(1, 2))
    s.append(Paragraph("<b>Anomaly-layer validation.</b> " + facts["rq3"], body))

    s.append(Paragraph("Limitations", h))
    for lim in facts["limitations"]:
        s.append(Paragraph("&bull;&nbsp;&nbsp;" + lim, body))
        s.append(Spacer(1, 1))

    s.append(Paragraph("Data lineage", h))
    s.append(kv_table(facts["lineage"]))

    s.append(Spacer(1, 8))
    s.append(Paragraph("For internal methodology and quality review.", small))

    doc.build(s)
    buf.seek(0)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_card.pdf")
    args = ap.parse_args()
    Path(args.out).write_bytes(build_model_card())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
