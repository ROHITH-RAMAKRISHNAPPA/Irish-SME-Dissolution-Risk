"""
Nexis news mention extraction.

Reads Nexis (LexisNexis) news DOCX exports and creates
a binary has_negative_news_mention flag per Irish company. Uses regex
pattern extraction for company names in legal notices, gazette structured
filings, and news articles, then fuzzy-matches the candidates against CRO
company names with RapidFuzz (token_sort_ratio >= 88).

The DOCX corpus is collected from LexisNexis Nexis News & Business
searches such as (Ireland and limited) and "winding up", "liquidator
appointed", and "winding up petition", typically over 2019-2024.

Output: data/processed/nexis_mentions.csv
    company_num, has_negative_news_mention, mention_count, source_files

Input: place all downloaded Nexis DOCX files in data/raw/nexis/
    (Nexis_winding_up_*.DOCX, Nexis_liquidator_*.DOCX, etc.)

Usage:
    pip install python-docx rapidfuzz --break-system-packages
    python src/extract_nexis.py
"""

import re
import sys
import zipfile
import pandas as pd
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import RAW_FILES, PROCESSED_FILES

# Install dependencies if needed
try:
    from docx import Document
    from rapidfuzz import process as rfprocess, fuzz
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "python-docx", "rapidfuzz", "--break-system-packages", "-q"])
    from docx import Document
    from rapidfuzz import process as rfprocess, fuzz

NEXIS_DIR        = RAW_FILES["nexis_dir"]
FUZZY_THRESHOLD  = 88   # token_sort_ratio threshold for company name matching
MIN_NAME_LEN     = 6    # ignore very short candidate names


def read_docx_text(path: Path) -> str:
    """Extract plain text from a DOCX file."""
    try:
        doc = Document(str(path))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  WARNING: Could not read {path.name}: {e}")
        return ""


# Company-suffix anchor for all extraction patterns
SUFFIX_PAT = r'(?:LIMITED|LTD|DAC|PLC|CLG|UNLIMITED|COMPANY|UC|TEORANTA)'

PATTERNS = [
    # High Court legal notices: "IN THE MATTER OF [NAME]"
    re.compile(
        r'IN THE MATTER OF\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE
    ),
    # Gazette structured notices
    re.compile(
        r'(?:Resolutions for Winding-up|Final Meetings|Notices to Creditors|'
        r'Meetings of Creditors|Annual Liquidation Meetings|Petitions to Wind Up \(Companies\)):\s*'
        r'([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE
    ),
    # News article patterns: "[NAME] wound up / being wound up / winds up"
    re.compile(
        r'([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')'
        r'\s+(?:wound up|being wound up|winds up|wound-up|in liquidation|'
        r'placed in liquidation|enters liquidation)',
        re.IGNORECASE
    ),
    # "liquidators appointed to [NAME]"
    re.compile(
        r'liquidators? appointed to\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE
    ),
    # "winding up of [NAME]"
    re.compile(
        r'winding up of\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE
    ),
]

# Legal-boilerplate phrases to filter out (not actual company names)
NOISE_PHRASES = {
    'THE COMPANIES ACT', 'THE HIGH COURT', 'THE MATTER', 'THE INSOLVENCY',
    'THE ABOVE', 'THE SAID', 'ALL CREDITORS', 'ANY CREDITOR', 'ANY PERSON',
    'PURSUANT TO', 'IN ACCORDANCE', 'BY ORDER', 'TAKE NOTICE',
}


def is_valid_name(name: str) -> bool:
    name_upper = name.upper().strip()
    if len(name_upper) < MIN_NAME_LEN:
        return False
    if any(noise in name_upper for noise in NOISE_PHRASES):
        return False
    if not re.search(SUFFIX_PAT, name_upper):
        return False
    return True


def extract_company_names(text: str) -> set:
    """Extract candidate company names from article text."""
    names = set()
    for pat in PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1).strip().upper().rstrip('.,;:)').strip()
            if is_valid_name(name):
                names.add(name)
    return names


def main():
    print("=" * 60)
    print("Nexis News Mention Extraction")
    print("Source: LexisNexis Nexis News & Business")
    print("=" * 60)

    if not NEXIS_DIR.exists():
        NEXIS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated: {NEXIS_DIR}")
        print("Place Nexis DOCX files here and re-run.")
        return

    # Deduplicate by filename (Windows case-insensitive filesystem matches both *.DOCX and *.docx)
    _seen = set()
    docx_files = []
    for _p in sorted(NEXIS_DIR.iterdir()):
        if _p.suffix.upper() == ".DOCX" and _p.name not in _seen:
            docx_files.append(_p)
            _seen.add(_p.name)
    docx_files = sorted(docx_files)
    if not docx_files:
        print(f"\nNo DOCX files found in {NEXIS_DIR}")
        print("Download from Nexis and place there.")
        return

    print(f"\nFound {len(docx_files)} Nexis files:")
    for f in docx_files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name:<50} {size_mb:.1f} MB")

    # Load CRO company names for matching
    print("\nLoading CRO company names...")
    cr = pd.read_csv(
        RAW_FILES["company_records"],
        usecols=["company_num", "company_name"],
        dtype={"company_num": str}
    )
    cr["company_num"]  = cr["company_num"].str.zfill(6)
    cr["name_upper"]   = cr["company_name"].str.upper().str.strip()
    cr = cr.dropna(subset=["name_upper"])
    cro_names = cr["name_upper"].tolist()
    print(f"  {len(cro_names):,} CRO company names loaded")

    name_to_num = dict(zip(cr["name_upper"], cr["company_num"]))

    # Process each DOCX file
    all_mentions: dict[str, set] = defaultdict(set)
    total_names_extracted = 0

    for docx_path in docx_files:
        print(f"\nProcessing {docx_path.name}...")
        text = read_docx_text(docx_path)
        if not text:
            continue

        candidates = extract_company_names(text)
        total_names_extracted += len(candidates)
        print(f"  Candidates extracted: {len(candidates)}")

        matched = 0
        for cand in candidates:
            result = rfprocess.extractOne(
                cand, cro_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD
            )
            if result:
                best_name, score, _ = result
                co_num = name_to_num[best_name]
                all_mentions[co_num].add(docx_path.name)
                matched += 1

        print(f"  Matched to CRO: {matched}")

    print(f"\nTotal candidates across all files: {total_names_extracted:,}")
    print(f"Unique CRO companies matched:      {len(all_mentions):,}")

    records = []
    for co_num, sources in all_mentions.items():
        records.append({
            "company_num":               co_num,
            "has_negative_news_mention": 1,
            "mention_count":             len(sources),
            "source_files":              "|".join(sorted(sources)),
        })

    result = pd.DataFrame(records)

    # Add 0-rows for all other companies (not mentioned)
    all_nums = set(cr["company_num"].tolist())
    mentioned = set(result["company_num"].tolist())
    zero_records = [
        {"company_num": n, "has_negative_news_mention": 0,
         "mention_count": 0, "source_files": ""}
        for n in all_nums - mentioned
    ]
    result = pd.concat([result, pd.DataFrame(zero_records)], ignore_index=True)
    result = result.sort_values("company_num").reset_index(drop=True)

    out_path = PROCESSED_FILES["nexis_mentions"]
    result.to_csv(out_path, index=False)

    print(f"\nOutput: {out_path}")
    print(f"  Total rows:       {len(result):,}")
    print(f"  With mention = 1: {result['has_negative_news_mention'].sum():,}")
    print(f"  Coverage:         {result['has_negative_news_mention'].mean():.2%}")


if __name__ == "__main__":
    main()
