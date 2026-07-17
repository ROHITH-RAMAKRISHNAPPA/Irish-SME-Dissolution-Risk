"""
Stage 2 NLP - Step 6: LLM entity-type classification across tiers.

Classifies each flagged company as special-purpose vehicle, trading business,
holding company, or uncertain: a distinction the filing metadata alone cannot
draw, using the model's world knowledge together with enriched CRO context.

The prompt is grounded in the structured CRO fields available for each company
(address, status, legal form, director count, NACE text description, and the
shared-address dissolution signal). When the name carries no clear signal and
the company is not recognisable, the classifier returns "uncertain" rather than
guess. The provider option selects the backend model; a frontier model
recognises named firms that a small local model does not.

Some companies are genuinely unclassifiable from any public field. These are
returned as "uncertain", which is the honest answer rather than a guess.

Two outputs: per-company classifications plus the cross-tier SPV-rate table.

Output: outputs/nlp/entity_types.csv
        outputs/nlp/entity_type_by_tier.csv
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
from config import OUTPUTS_DIR, PROCESSED_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
DEFAULT_OLLAMA = "llama3.1:8b"
DEFAULT_ANTHROPIC = "claude-sonnet-4-20250514"
DEFAULT_OPENAI = "gpt-4o-mini"

# Numeric facts to surface (only those present are used)
FACT_COLUMNS = [
    "ar_filed_count", "total_submissions", "annual_submission_rate",
    "director_change_count", "director_count", "company_age_years",
    "same_address_dissolution_count",
]
# Categorical/context fields to surface if present
CONTEXT_COLUMNS = [
    "company_type", "company_status", "company_address", "nace_notes",
]

VALID_TYPES = {"special_purpose_vehicle", "trading_business",
               "holding_company", "uncertain"}

CLASSIFY_PROMPT = """You classify Irish companies by entity type using the details \
below plus your own knowledge of real companies.

CRITICAL RULE: If the name does not clearly indicate the type AND you do not \
recognise the specific company, you MUST answer "uncertain". Do not guess \
"special_purpose_vehicle" from a generic or unfamiliar name. A real operating \
company (a bar, a pharma firm, an electrician, a design studio, a utility) is a \
trading_business even if it files little. Only classify as special_purpose_vehicle \
when the name or sector clearly indicates financing/securitisation/leasing/issuer \
activity (e.g. "Aircraft Leasing", "Funding", "Issuer", "ABS", "Capital DAC") or \
you recognise it as such.

Entity types:
- special_purpose_vehicle: financing / securitisation / leasing / issuer vehicle.
- holding_company: exists mainly to hold shares/assets in other companies.
- trading_business: a real operating business selling goods or services.
- uncertain: the information genuinely does not allow a confident call.

Company details:
{facts}

Respond ONLY with valid JSON:
{{
  "entity_type": "special_purpose_vehicle|holding_company|trading_business|uncertain",
  "recognised": true/false,
  "reason": "one sentence; if you answered uncertain, say what was missing"
}}"""


def check_ollama():
    try:
        requests.get(OLLAMA_TAGS, timeout=5).raise_for_status()
        return True
    except Exception as e:
        print(f"ERROR: Ollama not running on localhost:11434 ({e})")
        return False


def call_ollama(prompt, model, timeout=120):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "stream": False,
            "format": "json", "options": {"temperature": 0.0, "num_predict": 220},
        }, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"  Ollama call failed: {e}")
        return ""


def get_anthropic_client():
    """Read key from env or .env (same convention as app.py)."""
    key = ""
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY"):
        if os.environ.get(name, "").strip():
            key = os.environ[name].strip()
            break
    if not key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip().lstrip("export ").strip() in (
                            "ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY"):
                        key = v.strip().strip('"').strip("'")
                        break
    if not key:
        return None, "no API key in env or .env"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed (pip install anthropic)"
    try:
        return anthropic.Anthropic(api_key=key), "ready"
    except Exception as e:
        return None, f"client init failed: {e}"


def call_anthropic(client, prompt, model, timeout=60):
    try:
        msg = client.messages.create(
            model=model, max_tokens=220,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  Anthropic call failed: {e}")
        return ""


def get_openai_client():
    """Read OpenAI key from env or .env (same convention as app.py)."""
    key = ""
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        if os.environ.get(name, "").strip():
            key = os.environ[name].strip()
            break
    if not key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip().lstrip("export ").strip() in ("OPENAI_API_KEY", "OPENAI_KEY"):
                        key = v.strip().strip('"').strip("'")
                        break
    if not key:
        return None, "no API key in env or .env"
    try:
        import openai
    except ImportError:
        return None, "openai package not installed (pip install openai)"
    try:
        return openai.OpenAI(api_key=key), "ready"
    except Exception as e:
        return None, f"client init failed: {e}"


def call_openai(client, prompt, model, timeout=60, max_retries=5):
    """Call the API with exponential backoff. Returns the text on success, or
    None on hard failure (so the caller can skip the row instead of writing junk).
    Detects the daily-cap 429 and signals it distinctly so the run can stop clean."""
    import time as _t
    delay = 5
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=220, temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            # Daily request cap: retrying will not help until reset. Signal stop.
            if "requests per day" in msg or "RPD" in msg:
                print("  Daily request cap reached (RPD). Stopping cleanly.")
                return "__DAILY_CAP__"
            print(f"  OpenAI call failed (attempt {attempt+1}/{max_retries}): {msg[:80]}")
            if attempt < max_retries - 1:
                _t.sleep(delay)
                delay = min(delay * 2, 60)
    return None  # hard failure after retries


def parse(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
    try:
        d = json.loads(raw)
        et = str(d.get("entity_type", "uncertain")).strip().lower()
        if et not in VALID_TYPES:
            et = "uncertain"
        return et, bool(d.get("recognised", False)), str(d.get("reason", ""))[:300]
    except Exception:
        return "uncertain", False, "parse_error"


def build_facts(row, present_facts, present_ctx) -> str:
    lines = [f"Company name: {row.get('company_name','')}",
             f"NACE code: {row.get('nace_v2_code','')}",
             f"County: {row.get('county','')}"]
    for c in present_ctx:
        v = row.get(c)
        if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
            label = c.replace("_", " ")
            lines.append(f"{label}: {str(v)[:120]}")
    for c in present_facts:
        if pd.notna(row.get(c)):
            v = row[c]
            try:
                v = f"{float(v):.3g}"
            except (ValueError, TypeError):
                pass
            lines.append(f"{c}: {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="LLM entity-type classification (enriched)")
    ap.add_argument("--provider", choices=["ollama", "anthropic", "openai"], default="ollama")
    ap.add_argument("--model", default=None,
                    help="Model name (defaults: ollama=llama3.1:8b, "
                         "anthropic=claude-sonnet-4-20250514)")
    ap.add_argument("--per_tier", type=int, default=150,
                    help="Random sample per non-PRIORITY tier (ignored if --all)")
    ap.add_argument("--all", action="store_true",
                    help="Classify the entire cohort (all tiers, no sampling)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.model:
        model = args.model
    elif args.provider == "anthropic":
        model = DEFAULT_ANTHROPIC
    elif args.provider == "openai":
        model = DEFAULT_OPENAI
    else:
        model = DEFAULT_OLLAMA

    client = None
    if args.provider == "anthropic":
        client, status = get_anthropic_client()
        if client is None:
            sys.exit(f"ERROR: Anthropic unavailable: {status}")
        print(f"  Anthropic: {status}")
    elif args.provider == "openai":
        client, status = get_openai_client()
        if client is None:
            sys.exit(f"ERROR: OpenAI unavailable: {status}")
        print(f"  OpenAI: {status}")
    else:
        if not check_ollama():
            sys.exit(1)

    feat_path = PROCESSED_DIR / "prospective_final.csv"
    df = pd.read_csv(feat_path, low_memory=False)
    if "combined_risk_tier" not in df.columns:
        sys.exit("ERROR: combined_risk_tier not in prospective_final.csv")

    present_facts = [c for c in FACT_COLUMNS if c in df.columns]
    present_ctx = [c for c in CONTEXT_COLUMNS if c in df.columns]
    print(f"Entity-type classification (enriched)")
    print(f"  Provider/model: {args.provider} / {model}")
    print(f"  Context fields used: {present_ctx}")
    print(f"  Numeric fields used: {present_facts}")

    if args.all:
        work = df.reset_index(drop=True)
        print(f"  Scope: FULL cohort (all tiers)")
    else:
        frames = []
        for tier, sub in df.groupby("combined_risk_tier"):
            if tier == "PRIORITY":
                frames.append(sub)
            else:
                frames.append(sub.sample(n=min(args.per_tier, len(sub)),
                                         random_state=args.seed))
        work = pd.concat(frames).reset_index(drop=True)
        print(f"  Scope: PRIORITY + {args.per_tier}/tier sample")
    print(f"  To classify: {len(work):,} companies")
    print()

    out_path = NLP_DIR / "entity_types.csv"

    def num_key(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip().lstrip("0") or "0"

    done, results = set(), []
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        results = prev.to_dict("records")
        done = set(prev["company_num"].apply(num_key))
        print(f"Resume: {len(done):,} already done")

    start = time.time()
    pending = work[~work["company_num"].apply(num_key).isin(done)].reset_index(drop=True)
    processed = 0
    for i, row in pending.iterrows():
        prompt = CLASSIFY_PROMPT.format(facts=build_facts(row, present_facts, present_ctx))
        if args.provider == "anthropic":
            raw = call_anthropic(client, prompt, model)
        elif args.provider == "openai":
            raw = call_openai(client, prompt, model)
        else:
            raw = call_ollama(prompt, model)

        # Daily cap: save what we have and stop cleanly so --resume continues tomorrow.
        if raw == "__DAILY_CAP__":
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"\nDaily cap reached. Saved {len(results):,} rows. "
                  f"Re-run with --resume tomorrow to continue.")
            return

        # Hard failure after retries: skip this row entirely (do NOT write a fake
        # 'uncertain'). It stays un-done, so --resume will retry it next pass.
        if raw is None or raw == "":
            print(f"  Skipping {row['company_num']} (call failed after retries).")
            continue

        et, recog, reason = parse(raw)
        # A parse failure on a non-empty response is also a bad row: skip it.
        if reason == "parse_error":
            print(f"  Skipping {row['company_num']} (unparseable response).")
            continue

        results.append({
            "company_num": row["company_num"],
            "company_name": row["company_name"],
            "combined_risk_tier": row["combined_risk_tier"],
            "nace_2digit": str(row.get("nace_v2_code", ""))[:2],
            "entity_type": et,
            "recognised": recog,
            "reason": reason,
        })
        processed += 1
        if processed % 10 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)
            el = time.time() - start
            rate = processed / el if el else 0
            rem = (len(pending) - (i + 1)) / rate if rate else 0
            print(f"  [{i+1:4d}/{len(pending)}] {el/60:.1f}min, ~{rem/60:.1f}min remain")

    final = pd.DataFrame(results)
    final.to_csv(out_path, index=False)

    tab = (final.groupby("combined_risk_tier")["entity_type"]
           .value_counts().unstack(fill_value=0))
    tab["n"] = tab.sum(axis=1)
    if "special_purpose_vehicle" in tab.columns:
        tab["pct_spv"] = (100 * tab["special_purpose_vehicle"] / tab["n"]).round(1)
    if "uncertain" in tab.columns:
        tab["pct_uncertain"] = (100 * tab["uncertain"] / tab["n"]).round(1)
    tab.to_csv(NLP_DIR / "entity_type_by_tier.csv")

    print(f"\nDONE.  ({(time.time()-start)/60:.1f} min)")
    print(f"\nEntity type by tier:")
    print(tab.to_string())
    print(f"\nWrote: {out_path}")
    print(f"       {NLP_DIR / 'entity_type_by_tier.csv'}")


if __name__ == "__main__":
    main()
