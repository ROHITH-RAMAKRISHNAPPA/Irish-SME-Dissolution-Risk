"""
Stage 2 NLP - Step 1b: enrich the corpus with filing-sequence narratives.

Runs between nlp_01_corpus.py and nlp_03_topics.py. Reads the existing
corpus.csv, streams the raw CRO submissions history, and folds a plain-language
behavioural narrative of each company's ordered filing sequence into the
combined_text field. Downstream topic modelling and LLM extraction then operate
on filing-behaviour text rather than static structured descriptors alone.

The narrative is generated from filing metadata only (form type, received date,
registration status). No financial values, model scores, or information dated
after the observation window enters the text.

Structural handling forced by the raw source:
  - Each JSONL line is one company with a nested submissions list.
  - Filings are not date-ordered in the source; they are sorted before framing.
  - One filing event spans several document rows sharing a sub_num; events are
    collapsed to one per sub_num so a single return is not counted repeatedly.
  - Retro-scanned historic filings carry unreliable dates; the narrative details
    a recent window and summarises the deeper history.

Output: overwrites outputs/nlp/corpus.csv with an enriched combined_text
        (original structured text preserved in a new structured_only column)
        outputs/nlp/corpus_preview.csv refreshed
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"

# Years of detailed narration back from the observation date.
NARRATIVE_WINDOW_YEARS = 8

# Map raw sub_type_desc text to a compact behavioural event label.
# Matched by case-insensitive substring; first hit wins, so order matters.
EVENT_RULES = [
    ("annual return with accounts", "annual return with accounts"),
    ("annual return general", "annual return"),
    ("annual return short", "annual return"),
    ("annual return - no accounts", "annual return without accounts"),
    ("b1 annual return", "annual return"),
    ("annual return", "annual return"),
    ("change director or secretary", "director or secretary change"),
    ("change in dirs", "director or secretary change"),
    ("resignation of auditor", "auditor resignation"),
    ("registered office", "registered office change"),
    ("special resolution", "special resolution"),
    ("ordinary resolution", "ordinary resolution"),
    ("amended constitution", "constitution amendment"),
    ("company constitution", "constitution amendment"),
    ("conversion to ltd", "conversion to private limited"),
    ("mortgage", "charge registered"),
    ("charge", "charge registered"),
    ("receiver", "receiver appointed"),
    ("examinership", "examinership"),
    ("winding up", "winding up"),
    ("winding-up", "winding up"),
    ("strike off", "strike off notice"),
    ("strike-off", "strike off notice"),
    ("f8", "strike off notice"),
    ("high court order", "high court order"),
    ("nomination of new ard", "return date change"),
    ("nard", "return date change"),
    ("consolidation, division, conversion", "capital reorganisation"),
    ("capital duty", "capital duty statement"),
]


def event_label(sub_type_desc):
    low = str(sub_type_desc or "").lower()
    for needle, label in EVENT_RULES:
        if needle in low:
            return label
    return "other filing"


def parse_date(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d, (d.year < 1990 or d.year > 2027)
    except Exception:
        return None


def collapse_events(submissions):
    """One event per sub_num, earliest received date, ordered chronologically."""
    by_sub = {}
    for s in submissions:
        sub_num = s.get("sub_num")
        pd_ = parse_date(s.get("sub_received_date"))
        if pd_ is None:
            continue
        dt, suspect = pd_
        status = str(s.get("sub_status_desc") or "").lower()
        rec = by_sub.get(sub_num)
        if rec is None or dt < rec["dt"]:
            by_sub[sub_num] = {
                "dt": dt,
                "suspect": suspect,
                "label": event_label(s.get("sub_type_desc")),
                "rejected": "reject" in status,
            }
    return sorted(by_sub.values(), key=lambda r: r["dt"])


def build_narrative(company_name, events, obs_year):
    """Return (named_narrative, behaviour_text).

    named_narrative: human-readable story including the company name, for the
    LLM extraction step and dashboard display.
    behaviour_text: identical filing content with the company name and years
    removed, so downstream topic modelling clusters on filing-event patterns
    rather than on proper nouns in company names.
    """
    if not events:
        return (f"{company_name} has no filing history on record.",
                "no filing history")

    first_year = events[0]["dt"].year
    recent = [e for e in events
              if e["dt"].year >= obs_year - NARRATIVE_WINDOW_YEARS and not e["suspect"]]

    counts = defaultdict(int)
    rejected = 0
    for e in events:
        if e["suspect"]:
            continue
        counts[e["label"]] += 1
        if e["rejected"]:
            rejected += 1

    parts = [f"{company_name} has filed with the registry since {first_year}."]
    behaviour_bits = []

    ar_total = (counts.get("annual return", 0)
                + counts.get("annual return with accounts", 0)
                + counts.get("annual return without accounts", 0))
    dir_total = counts.get("director or secretary change", 0)
    summary_bits = []
    if ar_total:
        summary_bits.append(f"{ar_total} annual returns")
    if dir_total:
        summary_bits.append(f"{dir_total} director or secretary changes")
    for key in ["charge registered", "registered office change", "auditor resignation",
                "examinership", "winding up", "receiver appointed", "strike off notice",
                "conversion to private limited"]:
        if counts.get(key):
            n = counts[key]
            summary_bits.append(f"{n} {key}{'s' if n > 1 else ''}")
    if summary_bits:
        parts.append("Across its history: " + ", ".join(summary_bits) + ".")
        behaviour_bits.extend(summary_bits)
    if rejected:
        parts.append(f"{rejected} filing(s) were rejected.")
        behaviour_bits.append(f"{rejected} rejected filings")

    if recent:
        seq = []
        last_year = None
        behaviour_seq = []
        for e in recent:
            y = e["dt"].year
            tag = e["label"] + (" (rejected)" if e["rejected"] else "")
            if y != last_year:
                seq.append(f"{y}: {tag}")
                last_year = y
            else:
                seq.append(tag)
            behaviour_seq.append(tag)
        parts.append(
            f"In the {NARRATIVE_WINDOW_YEARS} years to {obs_year} the filing "
            f"sequence was: " + "; ".join(seq) + ".")
        behaviour_bits.append("recent sequence: " + "; ".join(behaviour_seq))
        gap = obs_year - recent[-1]["dt"].year
        if gap >= 2:
            parts.append(f"No filings recorded in the {gap} years before observation.")
            behaviour_bits.append(f"silent {gap} years before observation")
    else:
        parts.append(
            f"No filings in the {NARRATIVE_WINDOW_YEARS} years before observation; "
            f"the company appears dormant in the recent window.")
        behaviour_bits.append("dormant recent window, no recent filings")

    return " ".join(parts), ". ".join(behaviour_bits)


def norm(c):
    return str(c).strip().lstrip("0") or "0"


def main():
    ap = argparse.ArgumentParser(description="Enrich corpus with filing-sequence narratives")
    ap.add_argument(
        "--jsonl", required=True,
        help="path to cro_submissions_raw.jsonl (the raw per-filing history)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N matched companies (0 = all) for testing")
    args = ap.parse_args()

    corpus_path = NLP_DIR / "corpus.csv"
    if not corpus_path.exists():
        sys.exit(f"ERROR: {corpus_path} not found. Run nlp_01_corpus.py first.")

    corpus = pd.read_csv(corpus_path, low_memory=False)
    print(f"Stage 2 corpus enrichment (filing-sequence narratives)")
    print(f"  Corpus rows: {len(corpus):,}")

    corpus["cn_key"] = corpus["company_num"].apply(norm)
    wanted = set(corpus["cn_key"])

    # Observation year per company; falls back to a fixed year when absent.
    obs_years = {}
    if "obs_date" in corpus.columns:
        for _, r in corpus.iterrows():
            try:
                obs_years[r["cn_key"]] = int(str(r["obs_date"])[:4])
            except Exception:
                obs_years[r["cn_key"]] = 2024
    default_obs_year = 2024
    name_map = dict(zip(corpus["cn_key"], corpus["company_name"]))

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"ERROR: raw JSONL not found: {jsonl_path}")

    print(f"  Streaming {jsonl_path.name} ...")
    narrative_map = {}
    behaviour_map = {}
    events_map = {}
    seen = 0
    matched = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            seen += 1
            if seen % 100000 == 0:
                print(f"    scanned {seen:,} lines, matched {matched:,}")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cn = norm(obj.get("company_num"))
            if cn not in wanted:
                continue
            events = collapse_events(obj.get("submissions", []))
            obs_year = obs_years.get(cn, default_obs_year)
            narr, behav = build_narrative(name_map.get(cn, "This company"),
                                          events, obs_year)
            narrative_map[cn] = narr
            behaviour_map[cn] = behav
            events_map[cn] = len(events)
            matched += 1
            if args.limit and matched >= args.limit:
                print(f"    reached --limit {args.limit}, stopping scan")
                break

    print(f"  scanned {seen:,} lines, matched {matched:,} of {len(wanted):,} companies")

    # Preserve the original structured descriptor, prepend the filing narrative.
    corpus["structured_only"] = corpus["combined_text"]
    corpus["filing_narrative"] = corpus["cn_key"].map(narrative_map).fillna("")
    corpus["behaviour_text"] = corpus["cn_key"].map(behaviour_map).fillna("")
    corpus["n_filing_events"] = corpus["cn_key"].map(events_map).fillna(0).astype(int)

    def merge_text(r):
        narr = r["filing_narrative"]
        base = str(r["structured_only"])
        return (narr + " " + base).strip() if narr else base

    corpus["combined_text"] = corpus.apply(merge_text, axis=1)
    corpus = corpus.drop(columns=["cn_key"])

    corpus.to_csv(corpus_path, index=False)
    preview_cols = [c for c in ["company_num", "company_name", "combined_risk_tier",
                                "n_filing_events", "filing_narrative", "combined_text"]
                    if c in corpus.columns]
    corpus[preview_cols].head(100).to_csv(NLP_DIR / "corpus_preview.csv", index=False)

    # Report enrichment coverage by tier.
    print(f"\n=== ENRICHMENT COVERAGE BY TIER ===")
    enriched = corpus["filing_narrative"].str.len() > 0
    for t in ["PRIORITY", "DISSOLUTION_RISK", "BEHAVIORAL_ANOMALY", "LOW_CONCERN"]:
        sub = corpus[corpus["combined_risk_tier"] == t]
        if len(sub):
            n_enr = int((sub["filing_narrative"].str.len() > 0).sum())
            avg = int(sub.loc[sub["filing_narrative"].str.len() > 0, "combined_text"]
                      .str.len().mean()) if n_enr else 0
            print(f"  {t:<20} enriched {n_enr:>6,} / {len(sub):>6,}  avg_chars={avg}")

    print(f"\nWrote enriched corpus : {corpus_path}")
    print(f"Wrote preview         : {NLP_DIR / 'corpus_preview.csv'}")


if __name__ == "__main__":
    main()
