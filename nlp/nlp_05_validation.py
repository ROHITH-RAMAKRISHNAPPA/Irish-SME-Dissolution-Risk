"""
Stage 2 NLP - Step 4: cross-cohort validation.

For every company in the corpus (all tiers), cross-check whether it appears in:
  - Strike Off Listed   (CRO formally flagged for non-compliance)
  - Dissolved status    (CRO formal dissolution)
  - Dissolutions Register (post-Apr 2025 confirmed dissolutions)
  - High-confidence LLM-extracted distress signals (if nlp_03 has run)

This is qualitative validation of the model's flags beyond statistical AUC,
suitable for the Chapter 6 industry-application discussion. Because the corpus
now spans all tiers, the per-tier summary shows external-confirmation rates
for PRIORITY/DISSOLUTION_RISK/BEHAVIORAL_ANOMALY/LOW_CONCERN side by side.

Output: outputs/nlp/cohort_validation.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, RAW_DIR_CRO

NLP_DIR = OUTPUTS_DIR / "nlp"


def main():
    argparse.ArgumentParser(description="Stage 2 cross-cohort validation").parse_args()

    corpus_path = NLP_DIR / "corpus.csv"
    if not corpus_path.exists():
        sys.exit(f"ERROR: {corpus_path} not found. Run nlp_01_corpus.py first.")
    corpus = pd.read_csv(corpus_path, low_memory=False)

    print(f"Stage 2 cross-cohort validation")
    print(f"  Cohort: {len(corpus):,} companies (all tiers)")

    cr_path = RAW_DIR_CRO / "Company_Records.csv"
    cr = pd.read_csv(cr_path, low_memory=False,
                     usecols=["company_num", "company_status"])
    cr["is_strike_off_listed"] = cr["company_status"].str.contains(
        "Strike Off Listed", case=False, na=False)
    cr["is_dissolved_status"] = cr["company_status"].str.contains(
        "Dissolved", case=False, na=False)
    print(f"  Company Records loaded: {len(cr):,}")

    # A company can hold more than one status row in the register; collapse to a
    # single row per company (OR across status flags) so the left join stays
    # one-to-one with the cohort.
    cr = (cr.groupby("company_num", as_index=False)
            .agg(is_strike_off_listed=("is_strike_off_listed", "max"),
                 is_dissolved_status=("is_dissolved_status", "max")))

    diss_path = RAW_DIR_CRO / "Dissolutions_since_april_2025.csv"
    diss = pd.read_csv(diss_path, low_memory=False)
    diss["in_dissolutions_register"] = True
    diss = diss[["company_num", "in_dissolutions_register"]].drop_duplicates()
    print(f"  Dissolutions register loaded: {len(diss):,}")

    val = corpus[["company_num", "company_name", "combined_risk_tier", "risk_score"]].copy()
    val = val.merge(
        cr[["company_num", "is_strike_off_listed", "is_dissolved_status"]],
        on="company_num", how="left")
    val = val.merge(diss, on="company_num", how="left")
    for c in ["in_dissolutions_register", "is_strike_off_listed", "is_dissolved_status"]:
        val[c] = val[c].fillna(False)

    # External confirmation = appears in any CRO register signal. (Nexis news is
    # excluded: it names already-failed companies absent from the active cohort,
    # so it carries no valid signal for prospective companies.)
    val["any_external_confirm"] = (
        val["is_strike_off_listed"] |
        val["is_dissolved_status"] |
        val["in_dissolutions_register"]
    )

    llm_path = NLP_DIR / "llm_features.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        llm["llm_high_confidence"] = llm["confidence"].str.lower() == "high"
        llm["llm_signal_count"] = llm["distress_signals"].fillna("").apply(
            lambda s: len([x for x in str(s).split(";") if x.strip()]))
        val = val.merge(
            llm[["company_num", "llm_high_confidence", "llm_signal_count",
                 "audit_narrative"]],
            on="company_num", how="left")
        print(f"  LLM features joined: {(~val['llm_signal_count'].isna()).sum():,} "
              f"of {len(val):,} have LLM extraction")

    out_path = NLP_DIR / "cohort_validation.csv"
    val.to_csv(out_path, index=False)

    by_tier = val.groupby("combined_risk_tier").agg(
        n=("company_num", "count"),
        strike_off=("is_strike_off_listed", "sum"),
        dissolved=("is_dissolved_status", "sum"),
        diss_register=("in_dissolutions_register", "sum"),
        any_external=("any_external_confirm", "sum"),
    )
    by_tier["pct_external"] = (100 * by_tier["any_external"] / by_tier["n"]).round(1)

    print(f"\nSummary by tier:")
    print(by_tier.to_string())
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
