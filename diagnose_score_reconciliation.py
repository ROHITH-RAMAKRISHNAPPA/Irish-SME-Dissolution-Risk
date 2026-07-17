import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats

from src.config import PROCESSED_FILES, MODELS_DIR, TABLES_DIR, FEATURE_COLS

AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "audit" / "test_audit.csv"


def ks_stat(y_true, y_score):
    return stats.ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def gains(y_true, y_score, shares=(0.05, 0.10, 0.20)):
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()
    out = []
    for s in shares:
        k = int(round(len(y_sorted) * s))
        captured = y_sorted[:k].sum() / total_pos
        out.append({
            "portfolio_share": s,
            "dissolutions_captured": captured,
            "lift_over_random": captured / s,
        })
    return pd.DataFrame(out)


print("=" * 72)
print("SCORE RECONCILIATION DIAGNOSTIC")
print("=" * 72)

audit = pd.read_csv(AUDIT_PATH, low_memory=False)
if "split" in audit.columns:
    audit = audit[audit["split"] == "test"].copy()
y_audit = audit["label_modelled"].values.astype(int)
s_cal = audit["model_risk_score"].values.astype(float)

print(f"\ntest_audit.csv rows on test split : {len(audit):,}")
print(f"positive rate                     : {y_audit.mean():.6f}")
print(f"distinct values in model_risk_score: {len(np.unique(s_cal)):,}")

vc = pd.Series(s_cal).value_counts()
print(f"largest tied block                : {vc.iloc[0]:,} companies at one score")
print(f"share of rows in top 10 tied blocks: {vc.head(10).sum() / len(s_cal):.1%}")

print("\nCALIBRATED SCORE (what test_audit.csv holds)")
print(f"  AP  : {average_precision_score(y_audit, s_cal):.4f}")
print(f"  AUC : {roc_auc_score(y_audit, s_cal):.4f}")
print(f"  KS  : {ks_stat(y_audit, s_cal):.4f}")

test_df = pd.read_csv(PROCESSED_FILES["test_set"], low_memory=False)
X_test = test_df[FEATURE_COLS].values
y_test = test_df["label"].values.astype(int)

xgb_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
s_raw = xgb_model.predict_proba(X_test)[:, 1]

print("\nRAW SCORE (what Table 5.1 holds)")
print(f"  rows                            : {len(y_test):,}")
print(f"  positive rate                   : {y_test.mean():.6f}")
print(f"  distinct values                 : {len(np.unique(s_raw)):,}")
print(f"  AP  : {average_precision_score(y_test, s_raw):.4f}   Table 5.1 says 0.6298")
print(f"  AUC : {roc_auc_score(y_test, s_raw):.4f}   Table 5.1 says 0.9412")
print(f"  KS  : {ks_stat(y_test, s_raw):.4f}   Table 5.1 says 0.7400")

iso = joblib.load(MODELS_DIR / "isotonic_calibrator.joblib")
print(f"\nIsotonic calibrator threshold pairs: {len(iso.X_thresholds_):,}")

s_recal = iso.predict(s_raw)
print("\nRAW SCORE PUT THROUGH THE CALIBRATOR (reproduces the audit file)")
print(f"  distinct values                 : {len(np.unique(s_recal)):,}")
print(f"  AP  : {average_precision_score(y_test, s_recal):.4f}")
print(f"  AUC : {roc_auc_score(y_test, s_recal):.4f}")

spearman = stats.spearmanr(s_raw, s_recal).statistic
print(f"\nSpearman r, raw versus calibrated : {spearman:.6f}")
print("  If this is 1.0 the ordering is preserved and every AP gap above is ties alone.")

print("\n" + "=" * 72)
print("TRIAGE GAINS ON BOTH SCORES")
print("=" * 72)

g_cal = gains(y_audit, s_cal)
g_raw = gains(y_test, s_raw)

merged = g_raw.merge(g_cal, on="portfolio_share", suffixes=("_raw", "_calibrated"))
merged["captured_gap_pp"] = (
    merged["dissolutions_captured_raw"] - merged["dissolutions_captured_calibrated"]
) * 100

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print()
print(merged.to_string(index=False))

print("\ntriage_gains.csv on Drive currently reports:")
print("  0.05 -> 0.5860 (11.72x)")
print("  0.10 -> 0.7532 (7.53x)")
print("  0.20 -> 0.8813 (4.41x)")
print("Compare against the calibrated column above. They should match.")

out = merged.copy()
out.to_csv(TABLES_DIR / "score_reconciliation_diagnostic.csv", index=False)
print(f"\nWritten: {TABLES_DIR / 'score_reconciliation_diagnostic.csv'}")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
ap_raw = average_precision_score(y_test, s_raw)
ap_cal = average_precision_score(y_audit, s_cal)
if abs(ap_raw - 0.6298) < 0.005:
    print("Table 5.1 reproduces from the raw score. Chapter 5 is correct as written.")
else:
    print(f"Table 5.1 does NOT reproduce from the raw score ({ap_raw:.4f} vs 0.6298).")
    print("Stop here and work out why before touching anything else.")

if spearman > 0.9999:
    print("Ordering is identical. The AP gap is entirely tie-induced information loss")
    print(f"from binning {len(y_test):,} companies into {len(np.unique(s_recal)):,} distinct scores.")
else:
    print(f"Ordering is NOT preserved (Spearman {spearman:.6f}). That is unexpected for")
    print("isotonic regression and needs investigating separately.")

print(f"\nFigure 5.2 currently costs you {(ap_raw - ap_cal):.4f} AP by using the calibrated score.")
