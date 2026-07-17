"""
Rule-based likely-SPV classifier for the prospective cohort.

Labels each company as a likely special-purpose vehicle using three independent
structured indicators, with no dependence on the language model or news text:
  1. NACE sector in the financial / holding / leasing range (64, 65, 66, 77)
  2. Legal form is a Designated Activity Company (DAC)
  3. Company name contains a special-purpose-vehicle keyword

A company is a likely SPV if any one indicator fires. This is a display-only
label; it does not alter any trained model, score, or tier assignment.

When the language-model entity classification is available, this step also
reports the agreement between the two independent methods (rule-based here
versus model-based), which is a validation signal in its own right.

Output: outputs/nlp/prospective_spv_labelled.csv  (all companies, rule label)
        outputs/nlp/priority_spv_split.csv         (PRIORITY split Core vs SPV)
        outputs/nlp/spv_method_agreement.csv        (if entity_types.csv exists)
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, PROCESSED_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"

SPV_NACE_PREFIXES = ("64", "65", "66", "77")

# Name tokens characteristic of securitisation / leasing / financing vehicles.
# Matched case-insensitively on whole words.
SPV_NAME_TOKENS = (
    "funding", "issuer", "leasing", "capital", "aircraft", "abs",
    "finance", "financing", "securities", "securitisation", "securitization",
    "receivables", "investments", "holdings", "loan", "credit", "asset",
    "designated activity company", " dac",
)

# Roman-numeral series suffix (e.g. "... FUND II", "... III LIMITED") is a
# common SPV naming pattern for tranche vehicles.
ROMAN_SERIES = re.compile(r"\b(?:ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\b", re.IGNORECASE)


def name_hits_spv_keyword(name: str) -> bool:
    if not isinstance(name, str):
        return False
    low = name.lower()
    if any(tok in low for tok in SPV_NAME_TOKENS):
        return True
    if ROMAN_SERIES.search(low):
        return True
    return False


def nace_is_financial(code) -> bool:
    s = str(code)
    return s[:2] in SPV_NACE_PREFIXES


def legal_form_is_dac(company_type) -> bool:
    s = str(company_type).upper()
    return "DAC" in s or "DESIGNATED ACTIVITY" in s


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["spv_signal_nace"] = df["nace_v2_code"].apply(nace_is_financial)
    df["spv_signal_dac"] = df["company_type"].apply(legal_form_is_dac)
    df["spv_signal_name"] = df["company_name"].apply(name_hits_spv_keyword)

    df["is_likely_spv"] = (
        df["spv_signal_nace"] | df["spv_signal_dac"] | df["spv_signal_name"]
    )

    # Split-aware display label for the PRIORITY tier only.
    def priority_split(row):
        if row.get("combined_risk_tier") != "PRIORITY":
            return ""
        return "PRIORITY-SPV" if row["is_likely_spv"] else "PRIORITY-Core"

    df["priority_class"] = df.apply(priority_split, axis=1)
    return df



def norm_key(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip().lstrip("0") or "0"


def main():
    # Resolve the prospective file from the standard locations.
    path = NLP_DIR / "prospective_final.csv"
    if not path.exists():
        path = PROCESSED_DIR / "prospective_final.csv"
    if not path.exists():
        sys.exit("prospective_final.csv not found in outputs/nlp or data/processed.")

    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    needed = {"nace_v2_code", "company_type", "company_name", "combined_risk_tier"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns: {sorted(missing)}")

    out = classify(df)

    tier_counts = out["combined_risk_tier"].value_counts()
    prio = out[out["combined_risk_tier"] == "PRIORITY"].copy()
    n_prio = len(prio)
    n_spv = int(prio["is_likely_spv"].sum())
    n_core = n_prio - n_spv

    print("\n=== TIER TOTALS ===")
    for t in ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]:
        if t in tier_counts:
            print(f"  {t:<20} {tier_counts[t]:>7,}")

    print("\n=== PRIORITY SPV SPLIT (rule-based) ===")
    print(f"  PRIORITY total       : {n_prio}")
    print(f"  PRIORITY-Core (real) : {n_core}  ({100*n_core/max(n_prio,1):.1f}%)")
    print(f"  PRIORITY-SPV  (fin)  : {n_spv}  ({100*n_spv/max(n_prio,1):.1f}%)")

    print("\n=== SIGNAL BREAKDOWN (PRIORITY tier) ===")
    print(f"  NACE financial (64/65/66/77) : {int(prio['spv_signal_nace'].sum())}")
    print(f"  Legal form = DAC             : {int(prio['spv_signal_dac'].sum())}")
    print(f"  Name keyword / series        : {int(prio['spv_signal_name'].sum())}")
    extra = int((prio["is_likely_spv"] & ~prio["spv_signal_nace"]).sum())
    print(f"  Caught by DAC/name beyond NACE alone : {extra}")

    # Write outputs.
    labelled_path = NLP_DIR / "prospective_spv_labelled.csv"
    labelled_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(labelled_path, index=False)

    prio_cols = [c for c in [
        "company_num", "company_name", "company_type", "nace_v2_code",
        "combined_risk_tier", "dissolution_risk_score",
        "spv_signal_nace", "spv_signal_dac", "spv_signal_name",
        "is_likely_spv", "priority_class",
    ] if c in out.columns]
    split_path = NLP_DIR / "priority_spv_split.csv"
    prio.to_csv(split_path, columns=prio_cols, index=False)

    print(f"\nWrote labelled cohort : {labelled_path}")
    print(f"Wrote PRIORITY split  : {split_path}")

    # Agreement with the model-based entity classification, if available.
    entity_path = NLP_DIR / "entity_types.csv"
    if entity_path.exists():
        ent = pd.read_csv(entity_path, low_memory=False)
        if "entity_type" in ent.columns and "company_num" in ent.columns:
            out["_k"] = out["company_num"].apply(norm_key)
            ent["_k"] = ent["company_num"].apply(norm_key)
            ent["llm_is_spv"] = (ent["entity_type"].astype(str).str.lower()
                                 == "special_purpose_vehicle")
            merged = out[["_k", "is_likely_spv"]].merge(
                ent[["_k", "llm_is_spv"]], on="_k", how="inner")
            if len(merged):
                agree = int((merged["is_likely_spv"] == merged["llm_is_spv"]).sum())
                total = len(merged)
                both = int((merged["is_likely_spv"] & merged["llm_is_spv"]).sum())
                rule_only = int((merged["is_likely_spv"] & ~merged["llm_is_spv"]).sum())
                llm_only = int((~merged["is_likely_spv"] & merged["llm_is_spv"]).sum())
                neither = int((~merged["is_likely_spv"] & ~merged["llm_is_spv"]).sum())
                print("\n=== METHOD AGREEMENT (rule-based vs model-based) ===")
                print(f"  Companies compared     : {total:,}")
                print(f"  Agreement              : {agree:,}  ({100*agree/total:.1f}%)")
                print(f"  Both say SPV           : {both:,}")
                print(f"  Rule-only SPV          : {rule_only:,}")
                print(f"  Model-only SPV         : {llm_only:,}")
                print(f"  Neither                : {neither:,}")

                # Raw agreement is dominated by the non-SPV majority (a trivial
                # all-negative labeller would score higher), so report
                # chance-corrected and positive-class concordance instead.
                a, b, c, d = both, rule_only, llm_only, neither
                po = (a + d) / total
                pe = ((a + b) / total) * ((a + c) / total) + \
                     ((c + d) / total) * ((b + d) / total)
                kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
                jaccard = a / (a + b + c) if (a + b + c) else float("nan")
                pos_agree = 2 * a / (2 * a + b + c) if (2 * a + b + c) else float("nan")
                print(f"  Cohen's kappa          : {kappa:.3f}")
                print(f"  Jaccard (SPV overlap)  : {jaccard:.3f}")
                print(f"  Positive agreement     : {pos_agree:.3f}")

                pd.DataFrame([{
                    "compared": total, "agreement_pct": round(100*agree/total, 1),
                    "both_spv": both, "rule_only": rule_only,
                    "model_only": llm_only, "neither": neither,
                    "cohen_kappa": round(kappa, 3),
                    "jaccard": round(jaccard, 3),
                    "positive_agreement": round(pos_agree, 3),
                }]).to_csv(NLP_DIR / "spv_method_agreement.csv", index=False)
                print(f"  Wrote {NLP_DIR / 'spv_method_agreement.csv'}")
    else:
        print("\n(Model entity classification not found yet; run nlp_06 to enable "
              "the rule-vs-model agreement report.)")


if __name__ == "__main__":
    main()
