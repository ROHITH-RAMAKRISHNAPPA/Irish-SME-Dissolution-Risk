"""
Director dissolution cross-reference.

Computes director_dissolution_count and director_max_dissolutions per
company by cross-referencing FAME director names against CRO
Company_Records.csv. This avoids needing FAME UK or any external source.

The pipeline:
1. Read FAME_directors.xlsx and extract all director names per company.
2. Build a name -> [company_nums] lookup across all FAME companies, so we
   know every company each director serves at.
3. Read Company_Records.csv for dissolution status per company_num.
4. For each director, count how many of their companies are dissolved.
5. For each FAME company, aggregate across all its current directors into
   director_dissolution_count (total) and director_max_dissolutions (max
   for any single director).

Limitations: only covers the ~33,490 FAME-covered Irish companies;
directorships at non-FAME companies are not captured; dissolution status
is as of the CRO export date.

Output: data/processed/director_dissolution.csv
    company_num, director_dissolution_count, director_max_dissolutions

Usage:
    python src/extract_director_dissolution.py
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

DISSOLVED_STATUSES = {
    "Dissolved", "Dissolved (liquidation)", "Dissolved (bankruptcy)",
    "Dissolved (merger or take-over)", "Dissolved (demerger)",
    "In liquidation", "Bankruptcy", "Liquidation",
    "Ceased IRL", "Ceased", "Struck off",
}


# Xlsx reader: shared strings parsed per <si> element,
# self-closing empty cells stripped before regex matching
def read_xlsx(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            raw_ss = f.read().decode("utf-8")
        with z.open("xl/worksheets/sheet1.xml") as f:
            sheet = f.read().decode("utf-8")
    si_texts = re.findall(r'<si>(.*?)</si>', raw_ss, re.DOTALL)
    strings  = [''.join(re.findall(r'<t[^>]*>([^<]*)</t>', si)) for si in si_texts]
    headers  = {}
    rows     = []
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
            headers = {k: str(v).replace("\r\n"," ").replace("\n"," ").strip()
                       for k,v in cells.items()}
        else:
            rows.append({headers.get(k,k): v for k,v in cells.items()})
    return pd.DataFrame(rows)


def bvd_to_cro(bvd) -> str | None:
    s  = str(bvd).upper().strip()
    co = re.sub(r"^IE0*", "", s).strip()
    return co.zfill(6) if co.isdigit() else None


def split_field(val) -> list:
    if not val or str(val).strip() in ("", "nan", "None"):
        return []
    return [x.strip() for x in str(val).replace('\r\n', '\n').split('\n') if x.strip()]


def main():
    print("=" * 60)
    print("Director Dissolution Cross-Reference")
    print("Source: FAME directors x CRO Company_Records.csv")
    print("=" * 60)

    # Step 1: Load CRO company status
    print("\nStep 1: Loading CRO company status...")
    cr = pd.read_csv(RAW_FILES["company_records"],
                     usecols=["company_num", "company_status"],
                     dtype={"company_num": str})
    cr["company_num"] = cr["company_num"].astype(str).str.zfill(6)
    cr["is_dissolved"] = cr["company_status"].apply(
        lambda s: int(str(s).strip() in DISSOLVED_STATUSES or
                      "dissolved" in str(s).lower() or
                      "ceased" in str(s).lower() or
                      "liquidat" in str(s).lower())
    )
    dissolved_set = set(cr.loc[cr["is_dissolved"] == 1, "company_num"])
    print(f"  Total companies: {len(cr):,}")
    print(f"  Dissolved:       {len(dissolved_set):,} ({len(dissolved_set)/len(cr):.1%})")

    # Step 2: Load FAME directors
    print("\nStep 2: Loading FAME directors...")
    df = read_xlsx(RAW_FILES["fame_directors"])
    df["company_num"] = df["BvD ID number"].apply(bvd_to_cro)
    df = df.dropna(subset=["company_num"])
    print(f"  Companies with directors: {len(df):,}")

    name_col = "Director Full name"
    stat_col = "Director Current or previous"

    # Step 3: Build director -> companies lookup
    print("\nStep 3: Building director -> company mapping...")
    director_companies: dict[str, set] = {}

    for _, row in df.iterrows():
        co_num   = row["company_num"]
        names    = split_field(row.get(name_col, ""))
        statuses = split_field(row.get(stat_col, ""))

        for i, name in enumerate(names):
            name_clean = name.strip().upper()
            if len(name_clean) < 3:
                continue
            status = statuses[i].lower() if i < len(statuses) else ""
            if "current" in status:
                if name_clean not in director_companies:
                    director_companies[name_clean] = set()
                director_companies[name_clean].add(co_num)

    print(f"  Unique current directors:  {len(director_companies):,}")
    print(f"  Directors at 2+ companies: {sum(1 for v in director_companies.values() if len(v)>1):,}")
    print(f"  Directors at 5+ companies: {sum(1 for v in director_companies.values() if len(v)>5):,}")

    top = sorted(director_companies.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\n  Top nominee directors:")
    for name, cos in top:
        diss = len(cos & dissolved_set)
        print(f"    {name[:40]:<40} {len(cos):>4} companies, {diss:>3} dissolved")

    # Step 4: Compute dissolutions per director
    print("\nStep 4: Computing dissolution counts per director...")
    director_diss_count: dict[str, int] = {
        name: len(cos & dissolved_set)
        for name, cos in director_companies.items()
    }

    # Step 5: Aggregate to company level
    print("\nStep 5: Aggregating to company level...")
    records = []

    for _, row in df.iterrows():
        co_num   = row["company_num"]
        names    = split_field(row.get(name_col, ""))
        statuses = split_field(row.get(stat_col, ""))

        curr_diss = []
        for i, name in enumerate(names):
            name_clean = name.strip().upper()
            if len(name_clean) < 3:
                continue
            status = statuses[i].lower() if i < len(statuses) else ""
            if "current" in status:
                diss = director_diss_count.get(name_clean, 0)
                curr_diss.append(diss)

        records.append({
            "company_num":               co_num,
            "director_dissolution_count": sum(curr_diss),
            "director_max_dissolutions":  max(curr_diss) if curr_diss else 0,
        })

    result = pd.DataFrame(records)

    print(f"\nResults:")
    print(f"  Companies: {len(result):,}")
    print(f"  director_dissolution_count > 0:  {(result['director_dissolution_count']>0).sum():,}")
    print(f"  director_dissolution_count mean: {result['director_dissolution_count'].mean():.4f}")
    print(f"  director_max_dissolutions  > 0:  {(result['director_max_dissolutions']>0).sum():,}")
    print(f"  director_max_dissolutions  mean: {result['director_max_dissolutions'].mean():.4f}")

    out_path = PROCESSED_FILES["fame_directors"].parent / "director_dissolution.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
