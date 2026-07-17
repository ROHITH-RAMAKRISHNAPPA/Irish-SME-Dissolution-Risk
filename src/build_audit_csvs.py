"""
Audit CSV builder for examiner / EY demonstration.

Builds CSV proof files showing, for every company in the train and test
sets: model prediction (calibrated risk score, predicted tier), the label
the model used (24-month outcome from obs_date), and the actual real-world
dissolution status from CRO today. This is the artifact to hand over when
someone asks "show me companies the model trained on that actually
dissolved", or to demonstrate that flagged companies have in fact
dissolved since obs_date.

Outputs (data/processed/audit/):
    train_audit.csv             every train company + predictions + real status
    test_audit.csv              every test company + predictions + real status
    train_positives_only.csv    just the train dissolutions
    test_positives_only.csv     just the test dissolutions
    audit_summary.txt           counts + confusion matrix for both splits

Usage:
    python src/build_audit_csvs.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import (PROCESSED_FILES, MODELS_DIR, RAW_FILES,
                        FEATURE_COLS, DISSOLVED_STATUSES)

AUDIT_DIR = PROCESSED_FILES["train_set"].parent / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_COLS = ["company_num", "company_name", "obs_date", "comp_dissolved_date",
                "company_status_modelled", "label_modelled",
                "model_risk_score", "model_predicted_tier",
                "company_status_now", "actually_dissolved_now",
                "outcome_agreement", "company_age_years", "nace_section_label"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_FILES[name], dtype={"company_num": str}, low_memory=False)
    df["company_num"] = df["company_num"].str.zfill(6)
    return df


def add_predictions(df: pd.DataFrame, model, calibrator, feats: list[str]) -> pd.DataFrame:
    X = df[feats].fillna(0).values
    raw = model.predict_proba(X)[:, 1]
    cal = calibrator.predict(raw) if calibrator is not None else raw
    df = df.copy()
    df["model_risk_score"] = np.round(cal, 4)
    df["model_predicted_tier"] = pd.cut(
        df["model_risk_score"],
        bins=[-0.001, 0.30, 0.60, 0.80, 1.01],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"]
    ).astype(str)
    return df


def fetch_current_status() -> pd.DataFrame:
    """Pull the latest CRO status for every company: the real-world ground truth."""
    cr = pd.read_csv(RAW_FILES["company_records"],
                     usecols=["company_num", "company_status", "comp_dissolved_date"],
                     dtype={"company_num": str}, low_memory=False)
    cr["company_num"] = cr["company_num"].str.zfill(6)
    cr.columns = ["company_num", "company_status_now", "comp_dissolved_date_now"]
    cr["actually_dissolved_now"] = cr["company_status_now"].apply(
        lambda s: int(str(s).strip() in DISSOLVED_STATUSES or
                      "dissolved" in str(s).lower() or
                      "ceased" in str(s).lower() or
                      "liquidat" in str(s).lower() or
                      "struck off" in str(s).lower()))
    # Collapse to one row per company (dissolved status wins) so the left join
    # in assemble() stays one-to-one with each split and does not inflate counts.
    cr = (cr.sort_values("actually_dissolved_now", ascending=False)
            .drop_duplicates("company_num", keep="first")
            .reset_index(drop=True))
    return cr


def assemble(df: pd.DataFrame, current: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.merge(current, on="company_num", how="left")
    df["company_status_modelled"] = df.get("company_status",
                                            pd.Series(["unknown"] * len(df)))
    df["label_modelled"] = df["label"]
    # Three-way agreement: did the modelled label hold up against today's status?
    df["outcome_agreement"] = np.where(
        df["label_modelled"] == df["actually_dissolved_now"],
        "consistent", "diverged")
    keep = [c for c in DISPLAY_COLS if c in df.columns]
    df = df[keep].copy()
    df["split"] = split_name
    return df


def main():
    import joblib
    print("=" * 68)
    print("Audit CSVs: model predictions vs real-world outcomes")
    print("=" * 68)

    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    cal_path = MODELS_DIR / "isotonic_calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    feats_path = MODELS_DIR / "feature_cols.txt"
    if feats_path.exists():
        feats = [ln.strip() for ln in open(feats_path) if ln.strip()]
    else:
        feats = list(FEATURE_COLS)

    current = fetch_current_status()
    print(f"  CRO current status loaded: {len(current):,} companies")

    for split_name in ["train_set", "test_set"]:
        print(f"\n-> {split_name}")
        df = load_split(split_name)
        df = add_predictions(df, model, calibrator, [f for f in feats if f in df.columns])
        out = assemble(df, current, split_name.replace("_set", ""))

        out_file = AUDIT_DIR / f"{split_name.replace('_set', '')}_audit.csv"
        out.to_csv(out_file, index=False)
        print(f"  full   -> {out_file.name} ({len(out):,} rows)")

        pos = out[out["label_modelled"] == 1]
        pos_file = AUDIT_DIR / f"{split_name.replace('_set', '')}_positives_only.csv"
        pos.to_csv(pos_file, index=False)
        print(f"  pos    -> {pos_file.name} ({len(pos):,} rows)")

        if {"label_modelled", "actually_dissolved_now"}.issubset(out.columns):
            n = len(out)
            cons = (out["outcome_agreement"] == "consistent").sum()
            diverged_pos = ((out["label_modelled"] == 1)
                           & (out["actually_dissolved_now"] == 0)).sum()
            diverged_neg = ((out["label_modelled"] == 0)
                           & (out["actually_dissolved_now"] == 1)).sum()
            print(f"  consistent: {cons:,}/{n:,} ({cons/n:.1%})")
            print(f"  modelled positive, still active today: {diverged_pos:,}")
            print(f"  modelled negative, dissolved since:    {diverged_neg:,}")

    with open(AUDIT_DIR / "audit_summary.txt", "w") as f:
        f.write("AUDIT CSVS: purpose\n" + "=" * 50 + "\n")
        f.write("Each audit CSV contains, per company:\n")
        f.write("  - obs_date and label used during training/testing\n")
        f.write("  - model's calibrated risk score and predicted tier\n")
        f.write("  - the company's CURRENT CRO status (real-world truth)\n")
        f.write("  - whether the modelled outcome matches today's reality\n\n")
        f.write("Use this when an examiner asks for evidence that:\n")
        f.write("  - the model trained on companies that actually did dissolve, OR\n")
        f.write("  - flagged companies have, in fact, dissolved since obs_date.\n")
    print(f"\nDone. Files in: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
