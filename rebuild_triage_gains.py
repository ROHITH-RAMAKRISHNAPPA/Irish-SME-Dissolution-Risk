import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROCESSED_FILES, MODELS_DIR, TABLES_DIR, FEATURE_COLS

SHARES = (0.05, 0.10, 0.20)

test_df = pd.read_csv(PROCESSED_FILES["test_set"], low_memory=False)
X_test = test_df[FEATURE_COLS].values
y_test = test_df["label"].values.astype(int)

meta = pd.read_csv(MODELS_DIR / "winner_meta.csv")
winner = dict(zip(meta["key"], meta["value"]))["winner_name"]
model = joblib.load(MODELS_DIR / (winner.lower().replace(" ", "_") + "_model.joblib"))

score = model.predict_proba(X_test)[:, 1]

ap = average_precision_score(y_test, score)
auc = roc_auc_score(y_test, score)

print("=" * 72)
print("TRIAGE GAINS, REBUILT ON THE RAW STAGE 1 SCORE")
print("=" * 72)
print(f"\nModel      : {winner}")
print(f"Test rows  : {len(y_test):,}")
print(f"Base rate  : {y_test.mean():.4%}")
print(f"AP         : {ap:.4f}")
print(f"AUC        : {auc:.4f}")

if abs(ap - 0.6298) > 0.005:
    print("\nSTOP. This does not reproduce Table 5.1's 0.6298. Do not use this output.")
    sys.exit(1)

print("\nReconciles against Table 5.1. Proceeding.\n")

order = np.argsort(-score, kind="mergesort")
y_sorted = y_test[order]
total_pos = y_sorted.sum()

rows = []
for s in SHARES:
    k = int(round(len(y_sorted) * s))
    captured = y_sorted[:k].sum() / total_pos
    rows.append({
        "portfolio_share": s,
        "dissolutions_captured": captured,
        "lift_over_random": captured / s,
    })

gains = pd.DataFrame(rows)
gains.to_csv(TABLES_DIR / "triage_gains.csv", index=False)

print(gains.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

print("\n" + "=" * 72)
print("SENTENCES FOR SECTION 5.2")
print("=" * 72)

g = {r["portfolio_share"]: r for _, r in gains.iterrows()}
print(f"\n  reviewing the highest-scoring 10% of the test population recovers")
print(f"  {g[0.10]['dissolutions_captured']:.1%} of the companies that subsequently dissolved,")
print(f"  {g[0.10]['lift_over_random']:.1f} times what reviewing the same number of companies")
print(f"  in no particular order would return. At 5% the yield is")
print(f"  {g[0.05]['dissolutions_captured']:.1%}, a lift of {g[0.05]['lift_over_random']:.1f} times,")
print(f"  and at 20% it is {g[0.20]['dissolutions_captured']:.1%}.")

print(f"\nWritten: {TABLES_DIR / 'triage_gains.csv'}")
print("\nNext: re-run the fig_23 cell in 03_eda so the figure reads the new CSV,")
print("then replace Figure 5.2 in the document and update the three numbers above.")
