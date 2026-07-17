"""
Stage 2 NLP - Step 8: SHAP-vs-LLM concordance.

Measures whether each company's LLM narrative emphasises the same filing-behaviour
features that SHAP identifies as that company's top risk drivers. The language model
is grounded on a fixed set of SHAP-relevant behavioural features and is required to
quote at least two by name, so the test is not whether it rediscovers drivers from
raw text but whether, given those candidate features, its per-company emphasis tracks
the model's per-company SHAP ranking. High concordance indicates the narrative layer
is faithful to the model rather than generic, complementing the confabulation check.

Reads:  outputs/prospective_shap.csv   (per-company top_drivers_json)
        outputs/nlp/llm_features.csv   (distress_signals + audit_narrative)
Output: outputs/nlp/shap_llm_concordance.csv          (per-company)
        outputs/nlp/shap_llm_concordance_by_tier.csv  (tier summary)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"

# The behavioural features the language model was shown and could cite by name.
CITEABLE_FEATURES = [
    "ar_filed_count", "total_submissions", "annual_submission_rate",
    "director_change_count", "company_age_years", "submission_history_years",
    "name_change_count", "other_form_count",
]

TIERS = ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]


def parse_drivers(js):
    """Return [(feature, |shap|)] from a top_drivers_json cell, ranked by magnitude."""
    try:
        data = json.loads(js)
    except (TypeError, ValueError):
        return []
    out = []
    for x in data:
        if "feature" in x and "shap" in x:
            try:
                out.append((x["feature"], abs(float(x["shap"]))))
            except (TypeError, ValueError):
                continue
    return sorted(out, key=lambda t: t[1], reverse=True)


def top_k_features(drivers, k):
    return {f for f, _ in drivers[:k]}


def top_citeable_feature(drivers):
    """Highest-magnitude driver that the model could actually cite."""
    for f, _ in drivers:
        if f in CITEABLE_FEATURES:
            return f
    return None


def cited_features(text):
    low = str(text).lower()
    return {f for f in CITEABLE_FEATURES if f in low}


def main():
    shap_path = OUTPUTS_DIR / "prospective_shap.csv"
    llm_path = NLP_DIR / "llm_features.csv"
    for p in (shap_path, llm_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found.")

    shap = pd.read_csv(shap_path, low_memory=False)
    llm = pd.read_csv(llm_path, low_memory=False)
    print(f"SHAP-vs-LLM concordance")
    print(f"  SHAP rows: {len(shap):,} | LLM rows: {len(llm):,}")

    df = shap[["company_num", "combined_risk_tier", "top_drivers_json"]].merge(
        llm[["company_num", "distress_signals", "audit_narrative"]],
        on="company_num", how="inner")
    print(f"  Merged on company_num: {len(df):,}")

    records = []
    for _, r in df.iterrows():
        drivers = parse_drivers(r["top_drivers_json"])
        top5 = top_k_features(drivers, 5)
        top3 = top_k_features(drivers, 3)
        elig_top = top_citeable_feature(drivers)
        cited = cited_features(f"{r['distress_signals']} {r['audit_narrative']}")

        n_cited = len(cited)
        hits5 = len(cited & top5)
        records.append({
            "company_num": r["company_num"],
            "combined_risk_tier": r["combined_risk_tier"],
            "n_features_cited": n_cited,
            "hits_top5": hits5,
            "precision_top5": hits5 / n_cited if n_cited else np.nan,
            "cites_any_top3": int(bool(cited & top3)),
            "leads_with_top_driver": (int(elig_top in cited)
                                      if elig_top is not None else np.nan),
            "jaccard_top5": (len(cited & top5) / len(cited | top5)
                             if (cited | top5) else np.nan),
        })

    out = pd.DataFrame(records)
    out.to_csv(NLP_DIR / "shap_llm_concordance.csv", index=False)

    def summarise(frame):
        return {
            "n": len(frame),
            "mean_features_cited": round(frame["n_features_cited"].mean(), 2),
            "precision_top5": round(frame["precision_top5"].mean(), 3),
            "cites_any_top3": round(frame["cites_any_top3"].mean(), 3),
            "leads_with_top_driver": round(frame["leads_with_top_driver"].mean(), 3),
            "jaccard_top5": round(frame["jaccard_top5"].mean(), 3),
        }

    rows = [{"cohort": "OVERALL", **summarise(out)}]
    for t in TIERS:
        sub = out[out["combined_risk_tier"] == t]
        if len(sub):
            rows.append({"cohort": t, **summarise(sub)})
    by_tier = pd.DataFrame(rows)
    by_tier.to_csv(NLP_DIR / "shap_llm_concordance_by_tier.csv", index=False)

    print("\n=== CONCORDANCE (mean per company) ===")
    print(by_tier.to_string(index=False))
    print(f"\nWrote {NLP_DIR / 'shap_llm_concordance.csv'}")
    print(f"Wrote {NLP_DIR / 'shap_llm_concordance_by_tier.csv'}")


if __name__ == "__main__":
    main()
