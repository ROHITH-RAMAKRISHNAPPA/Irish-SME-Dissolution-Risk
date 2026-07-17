"""
Helix flat-file export for the Irish SME Dissolution Risk model.

Writes one row per scored company in a flat, machine-ingestible shape so the
scores can feed an existing anomaly-detection layer (such as a Helix-style
structured input rather than requiring a parallel platform. Each row carries the
company identifier, the risk tier and score, the top contributing filing-metadata
features (name and signed log-odds contribution), and the model-level validation
statistics as provenance columns.

Run:
    python build_helix_export.py
    python build_helix_export.py --shap outputs/prospective_shap.csv --out outputs/tables/helix_export.csv

Reads outputs/prospective_shap.csv (company_num, combined_risk_tier,
dissolution_risk_score, top_drivers_json). If data/processed/prospective_final.csv
is available it also attaches combined_risk_score and if_anomaly_score.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Model-level validation provenance (attached to every row so the receiving
# system records how the scores were validated). Update if the model is re-run.
PERMUTATION_AUC = 0.5297
PERMUTATION_NULL_AUC = 0.5080
PERMUTATION_P = "<0.0001"
MODEL_AVERAGE_PRECISION = 0.6298

TOP_N_FEATURES = 5


def _find(*candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def _top_drivers(cell, n):
    """Return a list of (feature, signed_contribution) from top_drivers_json."""
    try:
        drivers = json.loads(cell)
    except (TypeError, ValueError):
        return []
    out = []
    for d in drivers[:n]:
        feat = str(d.get("feature", "")).strip()
        try:
            shap = round(float(d.get("shap")), 4)
        except (TypeError, ValueError):
            shap = ""
        out.append((feat, shap))
    return out


def build_helix_export(shap_path, final_path=None, top_n=TOP_N_FEATURES):
    shap = pd.read_csv(shap_path, low_memory=False)

    base_cols = [c for c in ["company_num", "company_name", "combined_risk_tier",
                             "dissolution_risk_score"] if c in shap.columns]
    out = shap[base_cols].copy()

    # Attach combined risk score and anomaly score when the full file is present.
    if final_path is not None:
        keep = ["company_num", "combined_risk_score", "if_anomaly_score"]
        final = pd.read_csv(final_path, low_memory=False,
                            usecols=lambda c: c in keep)
        if "company_num" in final.columns:
            out = out.merge(final, on="company_num", how="left")

    # Expand the top contributing features into flat columns.
    driver_rows = shap["top_drivers_json"].apply(lambda c: _top_drivers(c, top_n)) \
        if "top_drivers_json" in shap.columns else None
    if driver_rows is not None:
        for i in range(top_n):
            out[f"driver{i + 1}_feature"] = driver_rows.apply(
                lambda lst: lst[i][0] if len(lst) > i else "")
            out[f"driver{i + 1}_contribution"] = driver_rows.apply(
                lambda lst: lst[i][1] if len(lst) > i else "")

    # Model-level validation provenance on every row.
    out["model_average_precision"] = MODEL_AVERAGE_PRECISION
    out["anomaly_permutation_auc"] = PERMUTATION_AUC
    out["anomaly_null_auc"] = PERMUTATION_NULL_AUC
    out["anomaly_permutation_p"] = PERMUTATION_P

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shap_path = args.shap or _find(ROOT / "outputs" / "prospective_shap.csv",
                                   ROOT / "prospective_shap.csv")
    if shap_path is None:
        raise SystemExit("Could not find prospective_shap.csv; pass --shap.")
    final_path = args.final or _find(ROOT / "data" / "processed" / "prospective_final.csv")
    out_path = Path(args.out) if args.out else (ROOT / "outputs" / "tables" / "helix_export.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_helix_export(shap_path, final_path)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df):,} rows, {len(df.columns)} columns)")
    if final_path is None:
        print("Note: prospective_final.csv not found, so combined_risk_score and "
              "if_anomaly_score were omitted. Pass --final to include them.")


if __name__ == "__main__":
    main()
