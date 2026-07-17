"""
Master extraction runner.

Runs all four extraction scripts (FAME, Orbis, director dissolution,
Nexis) in the correct order to produce the processed CSV files that
the notebook pipeline consumes. Run once before executing any notebook.

The CRO charges collection (01_collect_cro_submissions_all.py) is run
separately because it takes several hours; cro_charges.csv is joined
in NB02 once it exists. cro_submissions_summary.csv must already be
present in data/raw/01_CRO_Raw/ before NB02 runs.

Usage:
    python src/run_extracts.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROCESSED_FILES


# Runs a single extraction step and reports success or graceful failure
def run_step(step_num, name, fn):
    print(f"\nSTEP {step_num}: {name}")
    print("-" * 40)
    try:
        fn()
    except Exception as e:
        print(f"  WARNING: {name} failed: {e}")
        print(f"  Features from this source will default to 0 in NB02.")


def main():
    print("\n" + "=" * 60)
    print("Data Extraction Pipeline")
    print("=" * 60)
    print("Run ONCE before any notebook. All 4 steps must complete.")
    print("=" * 60)

    import src.extract_fame as m
    run_step(1, "FAME extraction (companies + directors)", m.main)

    import src.extract_orbis as m
    run_step(2, "Orbis extraction (ownership + financials + operations)", m.main)

    import src.extract_director_dissolution as m
    run_step(3, "Director dissolution cross-reference", m.main)

    import src.extract_nexis as m
    run_step(4, "Nexis news mention extraction", m.main)

    # Verify all expected processed files exist after the run
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY: expected processed files")
    print("=" * 60)

    expected = [
        ("fame_companies",       "fame_companies.csv"),
        ("fame_directors",       "fame_directors.csv"),
        ("orbis_ownership",      "orbis_ownership.csv"),
        ("orbis_financials",     "orbis_financials.csv"),
        ("orbis_operations",     "orbis_operations.csv"),
        ("director_dissolution", "director_dissolution.csv"),
        ("nexis_mentions",       "nexis_mentions.csv"),
    ]
    optional = [
        ("cro_charges", "cro_charges.csv"),
    ]

    all_ok = True
    for key, filename in expected:
        path = PROCESSED_FILES.get(key)
        if path is None:
            path = _ROOT / "data" / "processed" / filename
        if Path(path).exists():
            rows = sum(1 for _ in open(path)) - 1
            print(f"  {filename:<40} {rows:>7,} rows  OK")
        else:
            print(f"  {filename:<40}  NOT FOUND")
            all_ok = False

    print()
    for key, filename in optional:
        path = _ROOT / "data" / "processed" / filename
        if Path(path).exists():
            rows = sum(1 for _ in open(path)) - 1
            print(f"  {filename:<40} {rows:>7,} rows  OK (optional)")
        else:
            print(f"  {filename:<40}  not yet generated (run 01_collect_cro_submissions_all.py separately)")

    # Verify the standalone CRO submissions summary file exists in data/raw
    subs_path = _ROOT / "data" / "raw" / "01_CRO_Raw" / "cro_submissions_summary.csv"
    print()
    if subs_path.exists():
        mb = subs_path.stat().st_size / (1024 * 1024)
        print(f"  {'cro_submissions_summary.csv':<40} {mb:>7.1f} MB  OK")
    else:
        print(f"  {'cro_submissions_summary.csv':<40}  MISSING (needed for NB02)")
        print(f"       Run: python src/01_collect_cro_submissions_all.py --all-companies")

    print()
    if all_ok:
        print("All required extractions complete.")
        print("Run notebooks in order: NB00 -> NB01 -> NB02 -> NB03 -> NB04 -> NB05 -> NB06")
    else:
        print("WARNING: Some extractions failed. Fix before running notebooks.")


if __name__ == "__main__":
    main()
