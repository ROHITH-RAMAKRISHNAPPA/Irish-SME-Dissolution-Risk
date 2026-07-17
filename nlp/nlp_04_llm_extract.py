"""
Stage 2 NLP - Step 3: LLM extraction via Ollama (local, free)/gpt-4o-mini.

For each company in the chosen tier (PRIORITY by default), query Ollama/gpt-4o-mini to
extract structured distress signals from the combined text corpus.


Output: outputs/nlp/llm_features.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "phi3.5"
DEFAULT_OPENAI = "gpt-4o-mini"


def load_openai_key():
    """Read the OpenAI key from environment or a project-level .env file."""
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip().lstrip("export ").strip() in ("OPENAI_API_KEY", "OPENAI_KEY"):
                    return v.strip().strip('"').strip("'")
    return ""


def get_openai_client():
    key = load_openai_key()
    if not key:
        return None, "no OpenAI API key in environment or .env"
    try:
        import openai
    except ImportError:
        return None, "openai package not installed (pip install openai)"
    try:
        return openai.OpenAI(api_key=key), "ready"
    except Exception as e:
        return None, f"client init failed: {e}"


def call_openai(client, prompt, model, timeout=60):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  OpenAI call failed: {e}")
        return ""


EXTRACTION_PROMPT = """You are an audit risk analyst reviewing one Irish SME company \
flagged by a dissolution-risk model.

You are given verified filing facts below. These facts ARE the information - they \
are detailed and sufficient. Do NOT say information is missing, unavailable, or \
unclear; the numbers below tell you how the company behaves. Use them.

Verified facts for this company:
{facts}

Write your finding by interpreting the ACTUAL NUMBERS. You MUST quote the named \
feature values from the facts (for example "director_change_count of 8", \
"ar_filed_count of 12", "annual_submission_rate of 1.4"), not only the risk score:
- A high director_change_count relative to company_age_years signals governance churn.
- A high ar_filed_count / total_submissions means the company files actively (NOT silent).
- A low count means thin engagement.
Every distress signal must name at least one specific feature and its value from \
the facts (not the risk score alone). The narrative must quote at least two named \
feature values. Do not invent events (resignations, address changes) that are not \
in the facts - reason only from the counts given.

If the name and NACE indicate a special-purpose / financing / leasing / holding \
vehicle, say so AND note that its filing pattern may be sector-normal rather than \
distress - but still ground this in the numbers.

For audit_steps, give 3-5 concrete actions an auditor should take for THIS \
company, tailored to its specific signals (for example, if director_change_count \
is high, include verifying recent director changes; if charges are present, \
include reviewing the charge register). Each step must be a specific action, not \
a generic platitude.

Respond ONLY with valid JSON:
{{
  "distress_signals": ["<signal naming a specific feature and value, e.g. 'governance churn (director_change_count of 8 over company_age_years of 3)'>"],
  "audit_narrative": "2-3 sentences quoting at least two named feature values above. Never say information is unavailable.",
  "audit_steps": ["<specific action tailored to this company's signals>"],
  "confidence": "high if multiple feature values align, medium if one, low only if all counts are near zero"
}}"""


# SHAP-relevant feature columns to surface as grounded facts (only those present)
FACT_COLUMNS = [
    "ar_filed_count", "total_submissions", "annual_submission_rate",
    "director_change_count", "company_age_years", "submission_history_years",
    "name_change_count", "other_form_count",
]


def build_facts_block(corpus_row, feat_row) -> str:
    """Assemble a verified-facts block from real feature values. feat_row is the
    matching prospective_final.csv row (or None if unavailable). Behavioural
    feature values are listed first so the model reasons from them; the model
    score is placed last as context rather than the lead number."""
    lines = [
        f"Company name: {corpus_row.get('company_name', '')}",
        f"County: {corpus_row.get('county', '')}",
        f"NACE code: {corpus_row.get('nace_v2_code', '')}",
    ]
    if feat_row is not None:
        lines.append("Filing-behaviour feature values:")
        for col in FACT_COLUMNS:
            if col in feat_row and pd.notna(feat_row[col]):
                val = feat_row[col]
                try:
                    val = f"{float(val):.3g}"
                except (ValueError, TypeError):
                    pass
                lines.append(f"  {col}: {val}")
    lines.append(f"Model dissolution risk score (context): {corpus_row.get('risk_score', '')}")
    return "\n".join(lines)


def check_ollama_running(model_name=None):
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"ERROR: Ollama is not running on localhost:11434. ({e})")
        print("  Start it with:    ollama serve")
        print(f"  Then pull model:  ollama pull {model_name or DEFAULT_MODEL}")
        return False


def call_ollama(prompt: str, model: str, timeout: int = 300) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 500},
        }, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"  Ollama call failed: {e}")
        return ""


def parse_response(raw: str) -> dict:
    if not raw:
        return {"distress_signals": [], "audit_narrative": "",
                "audit_steps": [],
                "confidence": "low", "parse_error": "empty"}
    try:
        data = json.loads(raw)
        sig = data.get("distress_signals")
        steps = data.get("audit_steps")
        return {
            "distress_signals": sig if isinstance(sig, list) else [],
            "audit_narrative": str(data.get("audit_narrative", "")),
            "audit_steps": steps if isinstance(steps, list) else [],
            "confidence": str(data.get("confidence", "low")),
            "parse_error": "",
        }
    except json.JSONDecodeError as e:
        return {
            "distress_signals": [],
            "audit_narrative": raw[:500],
            "audit_steps": [],
            "confidence": "low",
            "parse_error": str(e)[:200],
        }


def main():
    ap = argparse.ArgumentParser(description="LLM extraction on Stage 2 corpus")
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama",
                    help="Inference backend (ollama=local, openai=API)")
    ap.add_argument("--tier", default="PRIORITY",
                    help="Tier to process (PRIORITY recommended for local runs)")
    ap.add_argument("--all_tiers", action="store_true",
                    help="Process every company in the corpus, ignoring --tier")
    ap.add_argument("--model", default=None,
                    help="Model name (defaults: ollama=phi3.5, openai=gpt-4o-mini)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of companies (for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip companies already in the output CSV")
    args = ap.parse_args()

    model = args.model or (DEFAULT_OPENAI if args.provider == "openai" else DEFAULT_MODEL)

    client = None
    if args.provider == "openai":
        client, status = get_openai_client()
        if client is None:
            sys.exit(f"ERROR: OpenAI unavailable: {status}")
        print(f"  OpenAI: {status}")
    else:
        if not check_ollama_running(model):
            sys.exit(1)

    corpus_path = NLP_DIR / "corpus.csv"
    if not corpus_path.exists():
        sys.exit(f"ERROR: corpus not found at {corpus_path}. "
                 f"Run nlp_01_corpus.py first.")

    df = pd.read_csv(corpus_path, low_memory=False)
    if args.all_tiers:
        df = df.reset_index(drop=True)
    else:
        df = df[df["combined_risk_tier"] == args.tier].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    # Load real feature values to ground the LLM (prevents invented signals)
    from config import PROCESSED_DIR
    feat_path = PROCESSED_DIR / "prospective_final.csv"
    feat_lookup = {}

    def num_key(v):
        # Normalize a company number to a plain integer string so int, float,
        # and zero-padded string representations all match on lookup.
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip().lstrip("0") or "0"

    if feat_path.exists():
        keep = ["company_num"] + FACT_COLUMNS
        feats = pd.read_csv(feat_path, low_memory=False)
        missing_cols = [c for c in FACT_COLUMNS if c not in feats.columns]
        if missing_cols:
            print(f"  WARNING: grounding file missing feature columns: {missing_cols}")
            print(f"  Available columns include: {[c for c in feats.columns if c in FACT_COLUMNS]}")
        keep = [c for c in keep if c in feats.columns]
        feats = feats[keep]
        feat_lookup = {num_key(r["company_num"]): r for _, r in feats.iterrows()}
        # Report how many grounding rows actually carry non-zero feature values,
        # so a broken join surfaces immediately instead of silently zero-filling.
        nonzero = sum(
            1 for r in feat_lookup.values()
            if any(pd.notna(r.get(c)) and float(r.get(c) or 0) != 0
                   for c in FACT_COLUMNS if c in r)
        )
        print(f"  Grounding facts loaded for {len(feat_lookup):,} companies "
              f"({nonzero:,} with non-zero feature values)")
    else:
        print("  WARNING: prospective_final.csv not found - facts will be name/county only")

    out_path = NLP_DIR / "llm_features.csv"
    done = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(existing["company_num"].astype(str))
        print(f"Resume mode: {len(done):,} companies already processed")

    print(f"Stage 2 LLM extraction")
    print(f"  Provider:  {args.provider}")
    print(f"  Scope:     {'ALL TIERS' if args.all_tiers else args.tier}")
    print(f"  Model:     {model}")
    print(f"  Companies: {len(df):,}  ({len(done):,} done, {len(df) - len(done):,} to do)")
    print()

    results = []
    if args.resume and out_path.exists():
        results = pd.read_csv(out_path).to_dict("records")

    start = time.time()
    pending = df[~df["company_num"].astype(str).isin(done)].reset_index(drop=True)

    for i, row in pending.iterrows():
        feat_row = feat_lookup.get(num_key(row["company_num"]))
        facts = build_facts_block(row, feat_row)
        prompt = EXTRACTION_PROMPT.format(facts=facts)
        raw = (call_openai(client, prompt, model) if args.provider == "openai"
               else call_ollama(prompt, model))
        parsed = parse_response(raw)

        # Confabulation guard: catch claims contradicted by the real values.
        narrative = parsed["audit_narrative"].lower()
        sig_text = " ".join(parsed["distress_signals"]).lower()
        text = narrative + " " + sig_text
        confab = []
        if feat_row is not None:
            dcc = feat_row.get("director_change_count")
            arc = feat_row.get("ar_filed_count")
            tot = feat_row.get("total_submissions")
            if ("resign" in text or "director_resigned" in text) and \
               (pd.notna(dcc) and float(dcc) == 0):
                confab.append("claims director resignation but director_change_count=0")
            # false-silence: claims silent/no filings while clearly filing
            says_silent = any(k in text for k in
                              ["filing_silent", "silent", "no recent filing",
                               "no filings", "not filed", "lack of recent filing",
                               "ceased filing", "stopped filing"])
            active = ((pd.notna(arc) and float(arc) >= 3) or
                      (pd.notna(tot) and float(tot) >= 6))
            if says_silent and active:
                confab.append(
                    f"claims filing silence but ar_filed_count="
                    f"{arc}/total_submissions={tot}")
        confab = "; ".join(confab)

        results.append({
            "company_num": row["company_num"],
            "company_name": row["company_name"],
            "combined_risk_tier": row["combined_risk_tier"],
            "model": model,
            "distress_signals": "; ".join(parsed["distress_signals"]),
            "audit_narrative": parsed["audit_narrative"],
            "audit_steps": "; ".join(parsed.get("audit_steps", [])),
            "confidence": parsed["confidence"],
            "confab_flag": confab,
            "parse_error": parsed["parse_error"],
        })

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(pending) - (i + 1)) / rate if rate > 0 else 0
        flag = " CONFAB" if confab else ""
        print(f"  [{i + 1:3d}/{len(pending)}] {str(row['company_name'])[:44]:44s} "
              f"signals={len(parsed['distress_signals'])} "
              f"conf={parsed['confidence']:6s}{flag} "
              f"({elapsed/60:.1f}min, ~{remaining/60:.1f}min remain)")

        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pd.DataFrame(results).to_csv(out_path, index=False)

    final = pd.DataFrame(results)
    n_confab = int((final["confab_flag"].fillna("") != "").sum()) if "confab_flag" in final else 0
    print(f"\nDONE.")
    print(f"  Total companies: {len(results):,}")
    print(f"  Total time:      {(time.time() - start) / 60:.1f} minutes")
    if n_confab:
        print(f"  {n_confab} narrative(s) flagged as possible confabulation "
              f"(see confab_flag column)")
    else:
        print(f"  No confabulation flags raised.")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
