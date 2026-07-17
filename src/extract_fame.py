"""
FAME data extraction.

Reads raw FAME Excel exports (Moody's / Bureau van Dijk)
and writes clean, model-ready CSVs to data/processed/. Produces two files:
fame_companies.csv (one row per company with accounts-date features) and
fame_directors.csv (one row per company with aggregated director features).

Both XLSX files are parsed via a custom zip/XML reader because the FAME
exports use rich-text runs in shared strings and self-closing empty cells
that break naive regex parsing; the reader compensates for both quirks.

Usage:
    python src/extract_fame.py
"""

import re
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date as _date_, timedelta as _td_

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import RAW_FILES, PROCESSED_FILES


# Custom xlsx reader: parses shared strings per <si> element (handles rich-text runs)
# and strips self-closing empty cells before regex matching (preserves column alignment)
def read_xlsx(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            raw_ss = f.read().decode("utf-8")
        with z.open("xl/worksheets/sheet1.xml") as f:
            sheet = f.read().decode("utf-8")

    si_texts = re.findall(r'<si>(.*?)</si>', raw_ss, re.DOTALL)
    strings = [
        ''.join(re.findall(r'<t[^>]*>([^<]*)</t>', si))
        for si in si_texts
    ]

    headers = {}
    rows = []
    for rnum, rc in re.findall(r'<row r="(\d+)"[^>]*>(.*?)</row>', sheet, re.DOTALL):
        rc_clean = re.sub(r'<c [^>]+/>', '', rc)

        cells = {}
        for m in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', rc_clean, re.DOTALL):
            col, attrs, content = m.group(1), m.group(2), m.group(3)
            v = re.search(r"<v>([^<]+)</v>", content)
            if v:
                raw = v.group(1)
                cells[col] = strings[int(raw)] if 't="s"' in attrs else raw

        if int(rnum) == 1:
            headers = {
                k: str(v).replace("\r\n", " ").replace("\n", " ").strip()
                for k, v in cells.items()
            }
        else:
            rows.append({headers.get(k, k): v for k, v in cells.items()})

    return pd.DataFrame(rows)


# Converts Excel date serial number to pandas Timestamp
def serial_to_ts(val) -> pd.Timestamp:
    if val is None or str(val).strip() in ("", "nan", "None"):
        return pd.NaT
    try:
        return pd.Timestamp(_date_(1899, 12, 30) + _td_(days=int(float(str(val)))))
    except (ValueError, TypeError, OverflowError):
        pass
    return pd.to_datetime(str(val), errors="coerce", dayfirst=True)


# Converts BvD ID (IE076941) to zero-padded 6-digit CRO number (076941)
def bvd_to_cro(bvd) -> str | None:
    s = str(bvd).upper().strip()
    co = re.sub(r"^IE0*", "", s).strip()
    return co.zfill(6) if co.isdigit() else None


# Splits a newline-delimited FAME cell into a list of values
def split_field(val) -> list:
    if not val or str(val).strip() in ("", "nan", "None"):
        return []
    return [x.strip() for x in str(val).replace('\r\n', '\n').split('\n') if x.strip()]


# Extracts company-level FAME features: filing patterns, overdue flags, NACE codes
def extract_fame_companies(path: Path, obs_date: str = "2022-12-31") -> pd.DataFrame:
    print(f"  Reading {path.name}...")
    df = read_xlsx(path)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    if "BvD ID number" not in df.columns:
        raise ValueError(f"BvD ID number column not found. Available: {list(df.columns)}")

    df["company_num"] = df["BvD ID number"].apply(bvd_to_cro)
    df = df.dropna(subset=["company_num"]).drop_duplicates("company_num")
    print(f"  CRO numbers extracted: {len(df):,}")

    obs_dt   = pd.Timestamp(obs_date)
    obs_year = obs_dt.year

    acct_col_map = {
        "n":  "Accounts date Last avail. yr",
        "n1": "Accounts date Year - 1",
        "n2": "Accounts date Year - 2",
        "n3": "Accounts date Year - 3",
    }
    acct_cols = {k: v for k, v in acct_col_map.items() if v in df.columns}
    print(f"  Accounts date columns found: {list(acct_cols.keys())}")

    yr1 = pd.Series(0, index=df.index)
    yr2 = pd.Series(0, index=df.index)
    for k, col in acct_cols.items():
        ts = df[col].apply(serial_to_ts)
        yr = ts.apply(lambda x: x.year if pd.notna(x) else None)
        yr1 += (yr == obs_year - 1).fillna(False).astype(int)
        yr2 += (yr == obs_year - 2).fillna(False).astype(int)

    df["n_filings_yr1"] = yr1.clip(upper=1)
    df["n_filings_yr2"] = yr2.clip(upper=1)

    def overdue_flag(col_name):
        if col_name not in df.columns:
            return pd.Series(0, index=df.index)
        ts = df[col_name].apply(serial_to_ts)
        return (ts.notna() & (ts < obs_dt)).astype(int)

    df["fame_accounts_overdue"] = overdue_flag("Accounts next due date")
    df["fame_ar_overdue"]       = overdue_flag("Return next due date")

    if "Date of last accounts filed" in df.columns:
        ts_last = df["Date of last accounts filed"].apply(serial_to_ts)
        df["fame_days_since_accounts"] = (
            (obs_dt - ts_last).dt.days.clip(lower=0).fillna(-1).astype(int)
        )
    else:
        df["fame_days_since_accounts"] = -1

    if "Woco" in df.columns:
        df["fame_in_worldcompliance"] = (
            df["Woco"].astype(str).str.strip().str.lower()
            .isin(["yes", "true", "1"]).astype(int)
        )
    else:
        df["fame_in_worldcompliance"] = 0

    if "All NACE Rev. 2 codes" in df.columns:
        df["fame_nace"] = df["All NACE Rev. 2 codes"].apply(
            lambda x: str(x).split("\n")[0].split(",")[0].strip()[:4]
            if x and str(x) not in ("nan", "None", "") else None
        )
    else:
        df["fame_nace"] = None

    df["fame_covered"] = 1

    result = df[[
        "company_num", "n_filings_yr1", "n_filings_yr2",
        "fame_accounts_overdue", "fame_ar_overdue", "fame_days_since_accounts",
        "fame_in_worldcompliance", "fame_covered", "fame_nace"
    ]].copy()

    print(f"  Output: {len(result):,} companies")
    for col in ["n_filings_yr1", "n_filings_yr2", "fame_ar_overdue", "fame_in_worldcompliance"]:
        print(f"    {col:<35} mean={result[col].mean():.4f}  nonzero={(result[col]>0).sum():,}")

    return result


# Extracts and aggregates director-level features to company level (one row per company)
def extract_fame_directors(path: Path) -> pd.DataFrame:
    print(f"  Reading {path.name}...")
    df = read_xlsx(path)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    if "BvD ID number" not in df.columns:
        raise ValueError(f"BvD ID column not found. Columns: {list(df.columns)}")

    df["company_num"] = df["BvD ID number"].apply(bvd_to_cro)
    df = df.dropna(subset=["company_num"]).drop_duplicates("company_num")
    print(f"  Companies with CRO number: {len(df):,}")

    port_col = "No of cos in which a current directorship is held"
    wc_col   = "Director In WorldCompliance"
    name_col = "Director Full name"
    stat_col = "Director Current or previous"

    print(f"  Portfolio column found: {port_col in df.columns}")
    print(f"  WC column found: {wc_col in df.columns}")

    records = []
    for _, row in df.iterrows():
        co_num    = row["company_num"]
        names     = split_field(row.get(name_col, ""))
        statuses  = split_field(row.get(stat_col, ""))
        portfolio = split_field(row.get(port_col, ""))
        wc_flags  = split_field(row.get(wc_col, ""))

        curr_idx = [i for i, s in enumerate(statuses) if "current" in s.lower()]
        n_current = len(curr_idx)

        curr_p = []
        for i in curr_idx:
            if i < len(portfolio):
                try:
                    curr_p.append(float(portfolio[i]))
                except (ValueError, TypeError):
                    pass

        curr_wc = [
            1 if i < len(wc_flags) and wc_flags[i].lower() in ("yes", "true", "1") else 0
            for i in curr_idx
        ]

        records.append({
            "company_num":                   co_num,
            "director_count":                n_current,
            "director_avg_portfolio_size":   np.mean(curr_p) if curr_p else 0.0,
            "director_worldcompliance_flag": int(any(w == 1 for w in curr_wc)),
            "director_back_to_back_flag":    int(any(p > 5 for p in curr_p)),
            "director_dissolution_count":    0,
            "director_max_dissolutions":     0,
        })

    result = pd.DataFrame(records)
    print(f"  Output: {len(result):,} companies")
    for col in ["director_count", "director_avg_portfolio_size", "director_worldcompliance_flag"]:
        print(f"    {col:<40} mean={result[col].mean():.4f}")

    return result


def main():
    print("=" * 60)
    print("FAME Feature Extraction")
    print("=" * 60)

    for key in ["fame_export", "fame_directors"]:
        if not RAW_FILES[key].exists():
            print(f"ERROR: {RAW_FILES[key]} not found.")
            return

    fame_co   = extract_fame_companies(RAW_FILES["fame_export"])
    fame_dirs = extract_fame_directors(RAW_FILES["fame_directors"])

    fame_co.to_csv(PROCESSED_FILES["fame_companies"],   index=False)
    fame_dirs.to_csv(PROCESSED_FILES["fame_directors"], index=False)

    print(f"\nOutputs written to data/processed/:")
    print(f"  fame_companies.csv: {len(fame_co):,} rows")
    print(f"  fame_directors.csv: {len(fame_dirs):,} rows")


if __name__ == "__main__":
    main()
