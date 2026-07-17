"""
Read-only comparison of the current isotonic calibrator against a per-block
Beta shrink toward the base rate.

Writes nothing. Touches no notebook. Run it, read the table, decide.

    python test_recalibration.py

The question it answers: does removing the calibration ceiling change any
company's Stage 1 band? If the answer is zero, nothing in Chapter 5 moves and
the only affected artefacts are the displayed probabilities.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from src.config import PROCESSED_FILES, MODELS_DIR, FEATURE_COLS

print("=" * 88)
print("RECALIBRATION TEST  (read-only)")
print("=" * 88)

model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
iso_current = joblib.load(MODELS_DIR / "isotonic_calibrator.joblib")

test = pd.read_csv(PROCESSED_FILES["test_set"], low_memory=False)
prosp = pd.read_csv(PROCESSED_FILES["prospective_set"], low_memory=False)

y_test = test["label"].values.astype(int)
raw_test = model.predict_proba(test[FEATURE_COLS].values)[:, 1]
raw_prosp = model.predict_proba(prosp[FEATURE_COLS].values)[:, 1]

# NB04 fits the calibrator on the even-indexed test rows.
cal_idx = np.arange(0, len(y_test), 2)
rt, yt = raw_test[cal_idx], y_test[cal_idx]
BASE = float(y_test.mean())
print(f"\ntest rows {len(y_test):,} | calibration fold {len(cal_idx):,} | "
      f"base rate {BASE:.4f} | prospective {len(prosp):,}")


def beta_shrink(iso, rt, yt, m, base):
    """Shrink each PAVA block toward the base rate using that block's own count,
    then re-run PAVA so the calibrator stays monotonic in the raw score.

    A block holding k positives out of n gets (k + m*base) / (n + m). A block of
    one company that happened to dissolve no longer returns 1.0; a block of two
    thousand that mostly did barely moves. m is the prior weight in companies.
    """
    blocks = iso.predict(rt)
    xs, ys = [], []
    for p in np.unique(blocks):
        sel = blocks == p
        n_bin, k = int(sel.sum()), int(yt[sel].sum())
        xs.append(rt[sel])
        ys.append(np.full(n_bin, (k + m * base) / (n_bin + m)))
    return IsotonicRegression(out_of_bounds="clip").fit(
        np.concatenate(xs), np.concatenate(ys))


def bands(cp):
    p95, p80 = np.percentile(cp, 95), np.percentile(cp, 80)
    return np.where(cp >= p95, "High", np.where(cp >= p80, "Medium", "Low"))


rows = []
cur_p = np.round(iso_current.predict(raw_prosp), 4)
cur_t = iso_current.predict(raw_test)
rows.append(("current", cur_p, cur_t))
for m in (5, 10, 25, 50, 100):
    b = beta_shrink(iso_current, rt, yt, m, BASE)
    rows.append((f"Beta shrink m={m}", np.round(b.predict(raw_prosp), 4),
                 b.predict(raw_test)))

print(f"\n{'method':<20}{'at 1.0':>8}{'at 0.0':>8}{'max':>9}{'min':>9}"
      f"{'distinct':>10}{'High':>8}{'Med':>8}{'test AP':>10}{'test AUC':>10}{'Brier':>9}")
print("-" * 88)
base_band = None
for name, cp, ct in rows:
    bd = bands(cp)
    if base_band is None:
        base_band = bd
    print(f"{name:<20}{int((cp >= .9995).sum()):>8,}{int((cp <= .0005).sum()):>8,}"
          f"{cp.max():>9.4f}{cp.min():>9.4f}{len(np.unique(cp)):>10,}"
          f"{int((bd == 'High').sum()):>8,}{int((bd == 'Medium').sum()):>8,}"
          f"{average_precision_score(y_test, ct):>10.4f}"
          f"{roc_auc_score(y_test, ct):>10.4f}{brier_score_loss(y_test, ct):>9.4f}")

print("\n" + "=" * 88)
print("THE DECIDING NUMBER: companies changing Stage 1 band")
print("=" * 88)
for name, cp, _ in rows[1:]:
    moved = int((bands(cp) != base_band).sum())
    flag = "SAFE" if moved == 0 else "NOT SAFE"
    print(f"  {name:<20}{moved:>8,} of {len(cp):,}   ({100*moved/len(cp):.3f}%)   {flag}")

print("""
If every row reads 0, the recalibration is a monotonic relabelling: the same
companies sit in the same bands, so every tier count, every RQ2 window, the
tier register confirmation, section 5.7 and Table 5.5.3's AUC are all unchanged.
Chapter 5 does not move. Only the displayed probability changes.

If any row is non-zero, stop. That option reorders companies and Chapter 5 would
have to be re-verified end to end.

Pick the smallest m that takes 'at 1.0' to zero without materially moving Brier.
""")
