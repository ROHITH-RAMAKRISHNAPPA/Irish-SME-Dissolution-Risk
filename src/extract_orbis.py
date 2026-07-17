"""
Orbis Europe data extraction.

Reads raw Orbis Europe Excel exports (Moody's Analytics Orbis Europe
VL/L+M database) and writes clean,
model-ready CSVs to data/processed/. Produces three files:
orbis_ownership.csv (ownership structure features),
orbis_financials.csv (financial ratio features including ebit_margin),
and orbis_operations.csv (operational performance features).

Company identifiers vary by file: ownership uses "National ID" (multi-
identifier: CRO, VAT, LEI), while financials and operations use "BvD ID
number" (format: IE076941). get_company_num() tries "National ID" first
and falls back to "BvD ID number"; both are converted to zero-padded
6-digit CRO numbers.

Usage:
    python src/extract_orbis.py
"""

import re
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

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


# Extracts a CRO number from a National ID cell that may contain multiple identifiers
def extract_cro_number(national_id) -> str | None:
    for part in str(national_id).replace("\r", "").split("\n"):
        part = part.strip()
        if part.isdigit() and 4 <= len(part) <= 7:
            return part.zfill(6)
        m = re.match(r"IECRO\.(\d+)", part)
        if m:
            return m.group(1).zfill(6)
    return None


# Converts BvD ID (IE076941) to zero-padded 6-digit CRO number
def bvd_to_cro(bvd) -> str | None:
    s  = str(bvd).upper().strip()
    co = re.sub(r"^IE0*", "", s).strip()
    return co.zfill(6) if co.isdigit() else None


# Returns a Series of CRO numbers, trying "National ID" first then falling back to "BvD ID number"
def get_company_num(df: pd.DataFrame) -> pd.Series:
    nat_col = next(
        (c for c in df.columns if c.strip().lower() == "national id"),
        None
    )
    if nat_col is not None:
        result = df[nat_col].apply(extract_cro_number)
        if result.notna().sum() > 0:
            print(f"  Using '{nat_col}' column for company matching")
            return result

    bvd_col = next(
        (c for c in df.columns
         if "bvd" in c.lower() and "id" in c.lower() and "number" in c.lower()),
        None
    )
    if bvd_col is None:
        bvd_col = next(
            (c for c in df.columns if "bvd" in c.lower() and "id" in c.lower()),
            None
        )
    if bvd_col is not None:
        print(f"  Using '{bvd_col}' column for company matching")
        return df[bvd_col].apply(bvd_to_cro)

    print(f"  WARNING: No company identifier column found. Columns: {list(df.columns[:10])}")
    return pd.Series(dtype=str, index=df.index)


# Parses Orbis numeric values which may contain commas, comparison operators, or N/A markers
def safe_float(value) -> float:
    s = str(value).replace(">", "").replace("<", "").strip()
    if s in ("", "n.a.", "n/a", "N/A", "nan", "None", "n.s."):
        return np.nan
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


# Finds a column whose name contains all given keywords (case-insensitive)
def get_col(df: pd.DataFrame, *keywords) -> str | None:
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keywords):
            return c
    return None


# Extracts ownership structure features: corporate/foreign ownership flags, GUO counts, subsidiary counts
def extract_ownership(path: Path) -> pd.DataFrame:
    print(f"  Reading {path.name}...")
    df = read_xlsx(path)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    df["company_num"] = get_company_num(df)
    df = df.dropna(subset=["company_num"]).drop_duplicates("company_num").copy()
    print(f"  CRO numbers extracted: {len(df):,}")
    guo_type_col    = get_col(df, "guo", "type")
    guo_country_col = get_col(df, "guo", "country iso")
    guo_wc_col      = get_col(df, "guo", "worldcompliance")
    guo_bvd_col     = get_col(df, "guo", "bvd id")
    subs_ult_col    = get_col(df, "ultimately")
    print(f"  GUO type col: {guo_type_col}")
    print(f"  GUO country col: {guo_country_col}")
    print(f"  GUO BvD ID col: {guo_bvd_col}")
    df["is_corporate_owned"] = (df[guo_type_col].astype(str).str.strip().str.lower() == "corporate").astype(int) if guo_type_col else 0
    df["is_foreign_owned"]  = ((df[guo_country_col].astype(str).str.strip().str.upper() != "IE") & df[guo_country_col].notna() & (df[guo_country_col].astype(str).str.strip() != "")).astype(int) if guo_country_col else 0
    df["guo_worldcompliance"] = (df[guo_wc_col].astype(str).str.strip().str.lower() == "yes").astype(int) if guo_wc_col else 0
    if guo_bvd_col:
        valid_guo  = df[df[guo_bvd_col].astype(str).str.strip() != ""]
        guo_counts = valid_guo[guo_bvd_col].value_counts()
        df["guo_irish_company_count"] = df[guo_bvd_col].map(guo_counts).fillna(0).astype(int)
    else:
        df["guo_irish_company_count"] = 0
    df["n_subsidiaries_ult"] = pd.to_numeric(df[subs_ult_col], errors="coerce").fillna(0).astype(int) if subs_ult_col else 0
    result = df[["company_num", "is_corporate_owned", "is_foreign_owned", "guo_worldcompliance", "guo_irish_company_count", "n_subsidiaries_ult"]].copy()
    print(f"  Output: {len(result):,} rows")
    for col in result.columns[1:]:
        print(f"    {col:<35} mean={result[col].mean():.4f}  nonzero={(result[col]>0).sum():,}")
    return result


# Extracts financial ratio features and derived distress flags (insolvency, illiquidity, loss-making, etc.)
def extract_financials(path):
    print(f"  Reading {path.name}...")
    df = read_xlsx(path)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    df["company_num"] = get_company_num(df)
    df = df.dropna(subset=["company_num"]).drop_duplicates("company_num").copy()
    print(f"  CRO numbers extracted: {len(df):,}")

    def gcol(*kw):
        candidates = [c for c in df.columns if all(k.lower() in c.lower() for k in kw)]
        last_yr = [c for c in candidates if "last avail" in c.lower()]
        return last_yr[0] if last_yr else (candidates[0] if candidates else None)

    def get_yrn_col(*nkw, n):
        return next(
            (c for c in df.columns
             if all(k.lower() in c.lower() for k in nkw)
             and f"year - {n}" in c.lower()),
            None
        )

    pl_col    = gcol("p/l before tax")
    pl1_col   = get_yrn_col("p/l before tax", n=1)
    pl2_col   = get_yrn_col("p/l before tax", n=2)
    ta_col    = gcol("total assets")
    sol_col   = gcol("solvency ratio")
    sol1_col  = get_yrn_col("solvency ratio", n=1)
    sol2_col  = get_yrn_col("solvency ratio", n=2)
    cr_col    = gcol("current ratio")
    em_col    = gcol("ebit margin")

    print(f"  Columns: pl={bool(pl_col)} ta={bool(ta_col)} sol={bool(sol_col)} cr={bool(cr_col)} ebit_margin={bool(em_col)}")

    def fcol(col):
        if not col or col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[col].apply(safe_float)

    pl0  = fcol(pl_col)
    pl1  = fcol(pl1_col)
    pl2  = fcol(pl2_col)
    sol0 = fcol(sol_col)
    sol1 = fcol(sol1_col)
    sol2 = fcol(sol2_col)
    cr0  = fcol(cr_col)
    ta0  = fcol(ta_col)
    em0  = fcol(em_col)

    df["pl_last_yr"]     = pl0
    df["total_assets"]   = ta0
    df["solvency_ratio"] = sol0
    df["current_ratio"]  = cr0

    df["is_loss_making"] = (pl0  < 0.0).fillna(False).astype(int)
    df["is_insolvent"]   = (sol0 < 8.0).fillna(False).astype(int)
    df["pl_declining"]   = ((pl0  < pl1)  & pl0.notna()  & pl1.notna()).astype(int)
    df["sol_declining"]  = ((sol0 < sol1) & sol0.notna() & sol1.notna()).astype(int)
    df["illiquid"]       = (cr0  < 1.0).fillna(False).astype(int)
    df["is_net_loss"]    = df["is_loss_making"]

    df["solvency_trend_3yr"] = (sol0 - sol2).where(sol0.notna() & sol2.notna())
    df["roaa"] = (pl0 / ta0.replace(0, np.nan)).where(ta0.notna() & pl0.notna())

    df["ebit_margin"] = em0

    loss_years = (
        (pl0 < 0).fillna(False).astype(int) +
        (pl1 < 0).fillna(False).astype(int) +
        (pl2 < 0).fillna(False).astype(int)
    )
    df["consecutive_loss_years"] = loss_years

    result = df[[
        "company_num", "pl_last_yr", "total_assets", "solvency_ratio", "current_ratio",
        "is_loss_making", "is_insolvent", "pl_declining", "sol_declining", "illiquid",
        "is_net_loss", "solvency_trend_3yr", "roaa", "ebit_margin", "consecutive_loss_years"
    ]].copy()

    print(f"  Output: {len(result):,} rows")
    for col in ["is_loss_making", "is_insolvent", "illiquid", "pl_declining", "sol_declining"]:
        print(f"    {col:<25} coverage={result[col].notna().sum():,}  mean={result[col].mean():.4f}")
    for col in ["is_net_loss", "solvency_trend_3yr", "roaa", "ebit_margin", "consecutive_loss_years"]:
        nn = result[col].notna().sum()
        print(f"    {col:<25} coverage={nn:,}  mean={result[col].mean():.4f}")

    return result


# Extracts operational performance features: revenue/EBIT trends, gearing, debt, employee counts
def extract_operations(path):
    print(f"  Reading {path.name}...")
    df = read_xlsx(path)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    df["company_num"] = get_company_num(df)
    df = df.dropna(subset=["company_num"]).drop_duplicates("company_num").copy()
    print(f"  CRO numbers extracted: {len(df):,}")

    def gcol(*kw):
        candidates = [c for c in df.columns if all(k.lower() in c.lower() for k in kw)]
        last = [c for c in candidates if "last avail" in c.lower()]
        return last[0] if last else (candidates[0] if candidates else None)

    def get_yr1(*kw):
        return next((c for c in df.columns if all(k.lower() in c.lower() for k in kw) and "year - 1" in c.lower()), None)

    def get_yr2(*kw):
        return next((c for c in df.columns if all(k.lower() in c.lower() for k in kw) and "year - 2" in c.lower()), None)

    rev0  = df.get(gcol("operating revenue"),     pd.Series(dtype=str)).apply(safe_float)
    rev1  = df.get(get_yr1("operating revenue"),  pd.Series(dtype=str)).apply(safe_float)
    rev2  = df.get(get_yr2("operating revenue"),  pd.Series(dtype=str)).apply(safe_float)
    ebit0 = df.get(gcol("operating profit"),      pd.Series(dtype=str)).apply(safe_float)
    ebit1 = df.get(get_yr1("operating profit"),   pd.Series(dtype=str)).apply(safe_float)
    gear0 = df.get(gcol("gearing"),               pd.Series(dtype=str)).apply(safe_float)
    ltd0  = df.get(gcol("long-term interest"),    pd.Series(dtype=str)).apply(safe_float)
    cp0   = df.get(gcol("credit period"),         pd.Series(dtype=str)).apply(safe_float)
    emp0  = df.get(gcol("number of employees"),   pd.Series(dtype=str)).apply(safe_float)
    emp1  = df.get(get_yr1("number of employees"),pd.Series(dtype=str)).apply(safe_float)

    df["revenue_declining"]     = ((rev0 < rev1)  & rev0.notna()  & rev1.notna()).astype(int)
    df["revenue_declining_2yr"] = ((rev0 < rev1)  & (rev1 < rev2) & rev0.notna() & rev1.notna() & rev2.notna()).astype(int)
    df["is_operating_loss"]     = (ebit0 < 0).fillna(False).astype(int)
    df["ebit_declining"]        = ((ebit0 < ebit1) & ebit0.notna() & ebit1.notna()).astype(int)
    df["highly_geared"]         = (gear0 > 200).fillna(False).astype(int)
    df["has_long_term_debt"]    = (ltd0 > 0).fillna(False).astype(int)
    df["slow_creditor_payment"] = (cp0 > 90).fillna(False).astype(int)
    df["employees_declining"]   = ((emp0 < emp1) & emp0.notna() & emp1.notna()).astype(int)
    df["ebitda_margin"]         = np.nan

    df["revenue_cagr_3yr"] = (
        ((rev0 / rev2.replace(0, np.nan)) ** 0.5 - 1)
        .where(rev0.notna() & rev2.notna() & (rev2 > 0))
    )

    result = df[[
        "company_num",
        "revenue_declining", "revenue_declining_2yr",
        "is_operating_loss", "ebit_declining",
        "highly_geared", "has_long_term_debt",
        "slow_creditor_payment", "employees_declining",
        "revenue_cagr_3yr", "ebitda_margin",
    ]].copy()

    print(f"  Output: {len(result):,} rows")
    for col in ["revenue_declining", "is_operating_loss", "highly_geared",
                "slow_creditor_payment", "employees_declining"]:
        print(f"    {col:<30} mean={result[col].mean():.4f}  nonzero={(result[col]>0).sum():,}")
    for col in ["revenue_cagr_3yr", "ebitda_margin"]:
        nn = result[col].notna().sum()
        print(f"    {col:<30} coverage={nn:,}  mean={result[col].mean():.4f}")

    return result


def main():
    print("=" * 62)
    print("Orbis Feature Extraction")
    print("Source: Moody's Orbis Europe VL/L+M")
    print("=" * 62)

    print("\nStep 1: Ownership features")
    ownership = extract_ownership(RAW_FILES["orbis_ownership_raw"])
    ownership.to_csv(PROCESSED_FILES["orbis_ownership"], index=False)
    print(f"  Saved: {PROCESSED_FILES['orbis_ownership']}")

    print("\nStep 2: Financial features (6-year export, 16 metrics, incl. ebit_margin)")
    financials = extract_financials(RAW_FILES["orbis_financials_raw"])
    financials.to_csv(PROCESSED_FILES["orbis_financials"], index=False)
    print(f"  Saved: {PROCESSED_FILES['orbis_financials']}")

    ops_raw = RAW_FILES.get("orbis_operations_raw")
    if ops_raw and ops_raw.exists():
        print("\nStep 3: Operational features (6-year export)")
        operations = extract_operations(ops_raw)
        operations.to_csv(PROCESSED_FILES["orbis_operations"], index=False)
        print(f"  Saved: {PROCESSED_FILES['orbis_operations']}")
        print(f"  orbis_operations.csv: {len(operations):,} rows")
    else:
        print("\nStep 3: orbis_operations_raw.xlsx not found: skipping")

    print("\n" + "=" * 62)
    print("Orbis extraction complete.")
    print(f"  orbis_ownership.csv: {len(ownership):,} rows")
    print(f"  orbis_financials.csv: {len(financials):,} rows (incl. ebit_margin)")


if __name__ == "__main__":
    main()
