"""
Stage 2 NLP - Step 1: build the per-company text corpus.

Nexis matching replicates the validated extract_nexis.py exactly:
  1. Extract candidate company names from adverse-event sentence structures
     ("IN THE MATTER OF [NAME] LIMITED", "Resolutions for Winding-up: [NAME]",
     "[NAME] LIMITED wound up / in liquidation", "liquidators appointed to
     [NAME]", "winding up of [NAME]"), each anchored on a company suffix.
  2. Filter legal boilerplate.
  3. Match each extracted candidate to the cohort with RapidFuzz
     token_sort_ratio >= config.FUZZY_MATCH_THRESH (88).
This is the same method that produced the 318 register-wide Nexis matches that
feed the Stage 1 has_negative_news_mention feature, so Stage 2 stays consistent
with Stage 1.

For every prospective company (all tiers by default) it then assembles:
  - the matched Nexis snippet(s), if any
  - structured CRO text (name, NACE descriptor, county, legal form, age)
  - a combined text field for downstream topic modelling and LLM extraction

Output: outputs/nlp/corpus.csv
        outputs/nlp/corpus_preview.csv   (first 100 rows for spot-check)
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from docx import Document
from rapidfuzz import process as rfprocess, fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR_NEXIS, FUZZY_MATCH_THRESH

NLP_DIR = OUTPUTS_DIR / "nlp"
NLP_DIR.mkdir(exist_ok=True, parents=True)

# Column names in prospective_final.csv
TIER_COL = "combined_risk_tier"
SCORE_COL = "dissolution_risk_score"

MIN_NAME_LEN = 6               # ignore very short candidate names
TOP_SNIPPETS_PER_COMPANY = 5

# Name-extraction logic: adverse-event company-name patterns.
SUFFIX_PAT = r'(?:LIMITED|LTD|DAC|PLC|CLG|UNLIMITED|COMPANY|UC|TEORANTA)'

PATTERNS = [
    # High Court legal notices: "IN THE MATTER OF [NAME]"
    re.compile(
        r'IN THE MATTER OF\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE,
    ),
    # Gazette structured notices
    re.compile(
        r'(?:Resolutions for Winding-up|Final Meetings|Notices to Creditors|'
        r'Meetings of Creditors|Annual Liquidation Meetings|Petitions to Wind Up \(Companies\)):\s*'
        r'([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE,
    ),
    # News article: "[NAME] wound up / being wound up / winds up / in liquidation"
    re.compile(
        r'([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')'
        r'\s+(?:wound up|being wound up|winds up|wound-up|in liquidation|'
        r'placed in liquidation|enters liquidation)',
        re.IGNORECASE,
    ),
    # "liquidators appointed to [NAME]"
    re.compile(
        r'liquidators? appointed to\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE,
    ),
    # "winding up of [NAME]"
    re.compile(
        r'winding up of\s+([A-Z][A-Z0-9\s\(\)&\-\.\',/]{3,80}?' + SUFFIX_PAT + r')',
        re.IGNORECASE,
    ),
]

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
    """Extract candidate company names from a block of article text."""
    names = set()
    for pat in PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1).strip().upper().rstrip('.,;:)').strip()
            if is_valid_name(name):
                names.add(name)
    return names
# -----------------------------------------------------------------------------

NACE_LABELS = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
}


def extract_candidates_with_snippets(nexis_dir: Path) -> dict:
    """Read every Nexis DOCX paragraph, extract candidate company names from the
    adverse-event structures, and map each candidate name to the snippets it
    appeared in. {candidate_name_upper: [(category, snippet), ...]}."""
    cand_snippets = defaultdict(list)
    for docx_path in sorted(nexis_dir.glob("Nexis_*.DOCX")):
        category = docx_path.stem.replace("Nexis_", "").rsplit("_", 1)[0]
        try:
            doc = Document(str(docx_path))
        except Exception as e:
            print(f"  WARNING: could not read {docx_path.name}: {e}")
            continue
        n_here = 0
        for p in doc.paragraphs:
            text = p.text.strip()
            if len(text) < MIN_NAME_LEN:
                continue
            for name in extract_company_names(text):
                cand_snippets[name].append((category, text[:500]))
                n_here += 1
        print(f"  loaded {docx_path.name} (category='{category}', "
              f"{n_here} candidate mentions)")
    return cand_snippets


def build_structured_text(row: pd.Series) -> str:
    name = str(row.get("company_name", "") or "")
    nace = str(row.get("nace_v2_code", "") or "")
    county = str(row.get("county", "") or "")
    ctype = str(row.get("company_type", "") or "")
    age = row.get("company_age_years", 0)

    nace_letter = nace[0] if nace and nace[0].isalpha() else ""
    nace_label = NACE_LABELS.get(nace_letter, "")

    parts = [f"Company: {name}"]
    if nace_label:
        parts.append(f"Sector: {nace_label} (NACE {nace})")
    if county and county.lower() != "nan":
        parts.append(f"County: {county}")
    if ctype and ctype.lower() != "nan":
        parts.append(f"Legal form: {ctype}")
    if pd.notna(age) and age:
        try:
            parts.append(f"Age at observation: {float(age):.1f} years")
        except (ValueError, TypeError):
            pass
    return ". ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser(description="Build Stage 2 NLP corpus")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="Risk tiers to include (default: all tiers)")
    ap.add_argument("--threshold", type=int, default=FUZZY_MATCH_THRESH,
                    help=f"token_sort_ratio threshold 0-100 (default {FUZZY_MATCH_THRESH})")
    ap.add_argument("--no_nexis", action="store_true",
                    help="Skip Nexis matching entirely (structured text only). "
                         "Recommended for the prospective cohort: winding-up news "
                         "names already-failed companies absent from the active "
                         "set, so matches are almost all false positives.")
    args = ap.parse_args()

    prospective_path = PROCESSED_DIR / "prospective_final.csv"
    if not prospective_path.exists():
        sys.exit(f"ERROR: not found: {prospective_path}")

    df = pd.read_csv(prospective_path, low_memory=False)
    print(f"Stage 2 NLP - Building corpus")
    print(f"  Loaded {len(df):,} prospective companies")

    if TIER_COL not in df.columns:
        sys.exit(f"ERROR: column '{TIER_COL}' not in prospective_final.csv. Run NB05 first.")

    if args.tiers:
        df = df[df[TIER_COL].isin(args.tiers)].reset_index(drop=True)
        print(f"  Tier filter {args.tiers}: {len(df):,} companies")
    else:
        df = df.reset_index(drop=True)
        print(f"  All tiers: {len(df):,} companies")

    row_matches = defaultdict(list)
    if args.no_nexis:
        # Nexis news names already-wound-up companies, which are absent from the
        # active prospective cohort, so fuzzy matching them produces near-100%
        # false positives (nearest-neighbour collisions on structural tokens like
        # PROPERTIES / INVESTMENTS). Structured-text-only corpus avoids that.
        print("  Nexis matching: DISABLED (--no_nexis); structured text only")
    else:
        print(f"  Nexis match: token_sort_ratio >= {args.threshold} on extracted candidates")

        # 1. Extract candidate names + their snippets from Nexis
        print(f"\nExtracting candidates from {RAW_DIR_NEXIS}...")
        cand_snippets = extract_candidates_with_snippets(RAW_DIR_NEXIS)
        candidates = list(cand_snippets.keys())
        print(f"Extracted {len(candidates):,} unique candidate company names")

        # 2. Match each candidate to the best company name in the cohort
        df["__name_upper"] = df["company_name"].astype(str).str.upper().str.strip()
        name_to_rows = defaultdict(list)
        for i, nm in enumerate(df["__name_upper"].tolist()):
            name_to_rows[nm].append(i)
        unique_names = list(name_to_rows.keys())

        print(f"\nMatching {len(candidates):,} candidates against "
              f"{len(unique_names):,} unique cohort names...")
        start = time.time()
        for j, cand in enumerate(candidates):
            result = rfprocess.extractOne(
                cand, unique_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=args.threshold,
            )
            if result:
                best_name, score, _ = result
                for ridx in name_to_rows[best_name]:
                    for (cat, snip) in cand_snippets[cand]:
                        row_matches[ridx].append((cat, snip, score))
            if (j + 1) % 1000 == 0:
                print(f"  matched {j + 1:,}/{len(candidates):,} candidates "
                      f"({time.time() - start:.0f}s)")

    # 3. Build corpus rows for every company
    print(f"\nAssembling corpus...")
    corpus_rows = []
    nexis_matched = 0
    for i, row in df.iterrows():
        matches = row_matches.get(i, [])
        # dedupe snippets, keep highest-scoring first
        seen = set()
        uniq = []
        for cat, snip, score in sorted(matches, key=lambda m: m[2], reverse=True):
            key = snip[:120]
            if key not in seen:
                seen.add(key)
                uniq.append((cat, snip, score))
        nexis_categories = "; ".join(sorted({c for c, _, _ in uniq}))
        nexis_snippet_count = len(uniq)
        nexis_combined = "\n---\n".join(s for _, s, _ in uniq[:TOP_SNIPPETS_PER_COMPANY])

        structured = build_structured_text(row)
        combined = structured
        if nexis_combined:
            combined += "\n\nNews mentions:\n" + nexis_combined
            nexis_matched += 1

        corpus_rows.append({
            "company_num": row.get("company_num"),
            "company_name": row.get("company_name"),
            "combined_risk_tier": row.get(TIER_COL),
            "risk_score": row.get(SCORE_COL, 0),
            "nace_v2_code": row.get("nace_v2_code", ""),
            "county": row.get("county", ""),
            "structured_text": structured,
            "nexis_categories": nexis_categories,
            "nexis_snippet_count": nexis_snippet_count,
            "nexis_text": nexis_combined,
            "combined_text": combined,
        })

    corpus_df = pd.DataFrame(corpus_rows)

    out_path = NLP_DIR / "corpus.csv"
    corpus_df.to_csv(out_path, index=False)

    csv_cols = ["company_num", "company_name", "combined_risk_tier", "risk_score",
                "nexis_categories", "nexis_snippet_count", "combined_text"]
    csv_path = NLP_DIR / "corpus_preview.csv"
    corpus_df[csv_cols].head(100).to_csv(csv_path, index=False)

    print(f"\nDONE.")
    print(f"  Corpus rows:    {len(corpus_df):,}")
    pct = 100 * nexis_matched / len(corpus_df) if len(corpus_df) else 0
    print(f"  Nexis-matched:  {nexis_matched:,} ({pct:.1f}%)")
    print(f"  By tier (matched / total):")
    for tier in ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]:
        sub = corpus_df[corpus_df["combined_risk_tier"] == tier]
        if len(sub):
            m = int((sub["nexis_snippet_count"] > 0).sum())
            print(f"    {tier:25s} {m:5,} / {len(sub):6,}  ({100*m/len(sub):.1f}%)")
    print(f"\nWrote:   {out_path}")
    print(f"Preview: {csv_path}")


if __name__ == "__main__":
    main()
