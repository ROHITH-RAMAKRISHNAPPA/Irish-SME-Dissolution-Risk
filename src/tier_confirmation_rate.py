"""
Table 5 columns C and D, sourced from the model output and the CRO register only.

For each operational risk tier (assigned by the supervised + isolation-forest
model, stored in prospective_final.csv), report the share of companies already
confirmed by the CRO register as struck off, dissolved, or appearing in the
post-April-2025 dissolutions register, plus the lift of each tier's rate over
the cohort-wide rate. No NLP / LLM inputs are used.

Drop this file in the project root (next to app.py) or in src/, then run:

    .\.venv\Scripts\python.exe tier_confirmation_rate.py
"""
import sys
from pathlib import Path
import pandas as pd

# Locate config.py (lives in src/) whether this script sits in root or src/
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE / "src", HERE.parent / "src"):
    if (cand / "config.py").exists():
        sys.path.insert(0, str(cand))
        break
from config import RAW_FILES, PROCESSED_FILES

# 1. Model output: company -> operational risk tier
tiers = pd.read_csv(PROCESSED_FILES["prospective_final"], low_memory=False,
                    usecols=["company_num", "combined_risk_tier"])
tiers["company_num"] = tiers["company_num"].astype(str)

# 2. CRO Company Records status (a company can have several status rows;
#    OR-aggregate to one row per company so the join stays one-to-one).
cr = pd.read_csv(RAW_FILES["company_records"], low_memory=False,
                 usecols=["company_num", "company_status"])
cr["is_strike_off_listed"] = cr["company_status"].str.contains("Strike Off Listed", case=False, na=False)
cr["is_dissolved_status"]  = cr["company_status"].str.contains("Dissolved",         case=False, na=False)
cr = (cr.groupby("company_num", as_index=False)
        .agg(is_strike_off_listed=("is_strike_off_listed", "max"),
             is_dissolved_status =("is_dissolved_status",  "max")))
cr["company_num"] = cr["company_num"].astype(str)

# 3. CRO Dissolutions register (post-April 2025 confirmed dissolutions)
diss = pd.read_csv(RAW_FILES["dissolutions"], low_memory=False, usecols=["company_num"])
diss["company_num"] = diss["company_num"].astype(str)
diss["in_dissolutions_register"] = True
diss = diss.drop_duplicates("company_num")

# 4. Left-join register facts onto the model tiers (cohort size unchanged)
df = tiers.merge(cr, on="company_num", how="left").merge(diss, on="company_num", how="left")
for c in ["is_strike_off_listed", "is_dissolved_status", "in_dissolutions_register"]:
    df[c] = df[c].fillna(False)
df["confirmed"] = df["is_strike_off_listed"] | df["is_dissolved_status"] | df["in_dissolutions_register"]

# 5. Per-tier rate and lift over the cohort-wide rate
overall = df["confirmed"].mean()
order = ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]
g = df.groupby("combined_risk_tier")["confirmed"].agg(n="count", confirmed="sum")
g["rate"] = g["confirmed"] / g["n"]
g["lift"] = g["rate"] / overall

print(f"Cohort size             : {len(df):,}")
print(f"Cohort confirmation rate: {overall*100:.2f}%\n")
print(f"{'Tier':<22}{'n':>8}{'confirmed':>11}{'rate':>9}{'lift':>8}")
print("-" * 58)
for t in order:
    if t in g.index:
        r = g.loc[t]
        print(f"{t:<22}{int(r['n']):>8,}{int(r['confirmed']):>11,}{r['rate']*100:>8.1f}%{r['lift']:>7.1f}x")
