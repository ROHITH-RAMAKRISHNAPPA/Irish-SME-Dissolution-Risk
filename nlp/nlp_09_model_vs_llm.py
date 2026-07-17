"""
Stage 2 NLP - Step 9: model-vs-LLM tier agreement.

Independent head-to-head between the supervised model and the language model.
Each company is shown only its filing-behaviour feature values, with the model's
tier, score, and SHAP withheld, and the language model must assign its own risk
tier (PRIORITY / DISSOLUTION_RISK / BEHAVIORAL_ANOMALY / LOW_CONCERN) from plain
definitions. The call is then compared against the model tier.

Two of the model tiers are mechanical: PRIORITY combines a high supervised score
with an anomaly flag, and BEHAVIORAL_ANOMALY is an anomaly flag on its own. Neither
can be reproduced by reasoning over features alone, so exact four-way agreement will
understate the language model. The result is therefore scored two ways: a four-way
confusion matrix with a linear-weighted kappa (an adjacent-tier call counts as
partial agreement), and a collapsed elevated-vs-low kappa as the honest headline.
This measures agreement, not accuracy: the supervised model is calibrated on observed
dissolution outcomes and the language model is not, so moderate agreement is itself
evidence that the supervised calibration carries signal the language model cannot
infer unaided.

Run:    python nlp_09_model_vs_llm.py --provider openai [--resume] [--limit N]
Score:  python nlp_09_model_vs_llm.py --score

Reads:  data/processed/prospective_final.csv
Output: outputs/nlp/model_vs_llm.csv            (per-company tier + reason)
        outputs/nlp/model_vs_llm_agreement.csv  (headline metrics)
        outputs/nlp/model_vs_llm_confusion4.csv (four-way matrix)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, PROCESSED_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"
DEFAULT_OPENAI = "gpt-4o-mini"

# Behavioural feature values shown to the language model. The model's own tier,
# risk score, and SHAP values are deliberately excluded so the call is independent.
FACT_COLUMNS = [
    "ar_filed_count", "total_submissions", "annual_submission_rate",
    "director_change_count", "company_age_years", "submission_history_years",
    "name_change_count", "other_form_count",
]

# Ordered from least to most severe, used for the weighted kappa.
TIERS = ["LOW_CONCERN", "BEHAVIORAL_ANOMALY", "DISSOLUTION_RISK", "PRIORITY"]

RISK_PROMPT = """You are assigning an Irish company to a risk tier from only the \
filing-behaviour details below. You do not have any model score; form your own judgement.

Assign exactly one tier:
- "PRIORITY": the most serious cases, showing both signs of dissolution risk and an \
unusual or irregular filing pattern that stands out from typical companies.
- "DISSOLUTION_RISK": clear signs of elevated dissolution risk from the filing behaviour, \
though the pattern is not otherwise unusual.
- "BEHAVIORAL_ANOMALY": an unusual or irregular filing pattern compared with typical \
companies, but weaker signs of outright dissolution risk.
- "LOW_CONCERN": a steady, proportionate filing history with no notable signs.

Reason only from the values given. A filing count very high or very low relative to \
company age, frequent director changes, or sparse engagement can indicate risk or \
irregularity; a stable, proportionate history indicates low concern.

Company details:
{facts}

Respond ONLY with valid JSON:
{{
  "risk_tier": "PRIORITY|DISSOLUTION_RISK|BEHAVIORAL_ANOMALY|LOW_CONCERN",
  "reason": "one sentence citing a specific filing-behaviour value"
}}"""


def get_openai_client():
    """Read the OpenAI key from environment or a project-level .env file."""
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


def call_openai(client, prompt, model, max_retries=5, rpd_wait=90, rpd_max_waits=6):
    """Call the API with exponential backoff.

    Transient errors are retried with growing delay. A requests-per-day signal is
    treated as a replenishing quota: the call waits rpd_wait seconds and retries, up
    to rpd_max_waits times, so a quota that frees up during the run is pushed through
    automatically. Only if it persists past all waits does the run stop cleanly for a
    later --resume. Returns text on success, None on hard failure (caller skips the
    row), or a daily-cap signal so the run stops clean."""
    delay, attempt, rpd_waits = 5, 0, 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=200, temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            if "requests per day" in msg or "RPD" in msg:
                if rpd_waits < rpd_max_waits:
                    rpd_waits += 1
                    print(f"  Daily-cap signal; waiting {rpd_wait}s then retrying "
                          f"({rpd_waits}/{rpd_max_waits})...")
                    time.sleep(rpd_wait)
                    continue
                print("  Daily cap persisted after waits. Stopping cleanly.")
                return "__DAILY_CAP__"
            attempt += 1
            print(f"  OpenAI call failed (attempt {attempt}/{max_retries}): {msg[:80]}")
            if attempt >= max_retries:
                return None
            time.sleep(delay)
            delay = min(delay * 2, 60)


def parse(raw):
    """Return (tier, reason) or (None, None) on an unusable response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
    try:
        d = json.loads(raw)
        t = str(d.get("risk_tier", "")).strip().upper().replace(" ", "_")
        if t not in TIERS:
            low = t.lower()
            if "priorit" in low:
                t = "PRIORITY"
            elif "dissol" in low:
                t = "DISSOLUTION_RISK"
            elif "anomal" in low or "behav" in low:
                t = "BEHAVIORAL_ANOMALY"
            elif "low" in low:
                t = "LOW_CONCERN"
            else:
                return None, None
        return t, str(d.get("reason", ""))[:300]
    except Exception:
        return None, None


def build_facts(row, present_facts) -> str:
    lines = [f"Company name: {row.get('company_name', '')}",
             f"NACE code: {row.get('nace_v2_code', '')}",
             f"County: {row.get('county', '')}"]
    for c in present_facts:
        if pd.notna(row.get(c)):
            v = row[c]
            try:
                v = f"{float(v):.3g}"
            except (ValueError, TypeError):
                pass
            lines.append(f"{c}: {v}")
    return "\n".join(lines)


def num_key(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip().lstrip("0") or "0"


def _kappa(matrix, weights):
    """General Cohen's kappa given an observed count matrix and a weight matrix
    (1.0 = full agreement, 0.0 = full disagreement)."""
    n = matrix.values.sum()
    if not n:
        return float("nan")
    row = matrix.sum(axis=1).values
    col = matrix.sum(axis=0).values
    po = pe = 0.0
    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
            po += weights[i][j] * matrix.iat[i, j] / n
            pe += weights[i][j] * (row[i] / n) * (col[j] / n)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def score():
    """Compare the language model's tier calls against the model tiers, two ways."""
    path = NLP_DIR / "model_vs_llm.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run the classification first.")
    df = pd.read_csv(path)
    df = df[df["llm_tier"].isin(TIERS)].copy()
    k = len(TIERS)

    # Four-way confusion matrix (rows = model, cols = LLM), fixed tier order.
    ct = (pd.crosstab(df["combined_risk_tier"], df["llm_tier"])
            .reindex(index=TIERS, columns=TIERS, fill_value=0))
    ct.to_csv(NLP_DIR / "model_vs_llm_confusion4.csv")

    n = int(ct.values.sum())
    exact = sum(ct.iat[i, i] for i in range(k)) / n if n else float("nan")

    ident = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    linw = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    kappa_unw = _kappa(ct, ident)
    kappa_lin = _kappa(ct, linw)

    # Collapsed elevated-vs-low (LOW_CONCERN = low, every other tier = elevated).
    def binz(t):
        return "low" if t == "LOW_CONCERN" else "elevated"
    b_model = df["combined_risk_tier"].map(binz)
    b_llm = df["llm_tier"].map(binz)
    cats = ["elevated", "low"]
    ct2 = (pd.crosstab(b_model, b_llm).reindex(index=cats, columns=cats, fill_value=0))
    a, bb = int(ct2.loc["elevated", "elevated"]), int(ct2.loc["elevated", "low"])
    c, d = int(ct2.loc["low", "elevated"]), int(ct2.loc["low", "low"])
    n2 = a + bb + c + d
    po2 = (a + d) / n2 if n2 else float("nan")
    pe2 = (((a + bb) / n2) * ((a + c) / n2) + ((c + d) / n2) * ((bb + d) / n2)) if n2 else float("nan")
    kappa2 = (po2 - pe2) / (1 - pe2) if (n2 and (1 - pe2)) else float("nan")

    print("\n=== MODEL vs LLM (four-way) ===")
    print(f"  Companies scored : {n:,}")
    print("  Confusion matrix (rows = model, cols = LLM):")
    print(ct.to_string())
    print(f"\n  Exact match           : {100*exact:.1f}%")
    print(f"  Cohen's kappa (unweighted)      : {kappa_unw:.3f}")
    print(f"  Cohen's kappa (linear-weighted) : {kappa_lin:.3f}")
    print("\n=== Collapsed elevated-vs-low (headline) ===")
    print(ct2.to_string())
    print(f"  Agreement        : {100*po2:.1f}%")
    print(f"  Cohen's kappa    : {kappa2:.3f}")
    print(f"  Model-elevated, LLM-low : {bb:,}   Model-low, LLM-elevated : {c:,}")

    pd.DataFrame([{
        "n": n,
        "four_way_exact_pct": round(100 * exact, 1),
        "kappa_unweighted": round(kappa_unw, 3),
        "kappa_linear_weighted": round(kappa_lin, 3),
        "binary_n": n2,
        "binary_agreement_pct": round(100 * po2, 1),
        "binary_kappa": round(kappa2, 3),
        "both_elevated": a, "model_elevated_llm_low": bb,
        "model_low_llm_elevated": c, "both_low": d,
    }]).to_csv(NLP_DIR / "model_vs_llm_agreement.csv", index=False)
    print(f"\nWrote {NLP_DIR / 'model_vs_llm_agreement.csv'}")
    print(f"Wrote {NLP_DIR / 'model_vs_llm_confusion4.csv'}")


def main():
    ap = argparse.ArgumentParser(description="Model-vs-LLM tier agreement (four-way)")
    ap.add_argument("--provider", choices=["openai"], default="openai")
    ap.add_argument("--model", default=DEFAULT_OPENAI)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process only N companies (test run)")
    ap.add_argument("--score", action="store_true", help="Score an existing model_vs_llm.csv")
    args = ap.parse_args()

    if args.score:
        score()
        return

    client, status = get_openai_client()
    if client is None:
        sys.exit(f"ERROR: OpenAI unavailable: {status}")
    print(f"  OpenAI: {status}")

    df = pd.read_csv(PROCESSED_DIR / "prospective_final.csv", low_memory=False)
    if "combined_risk_tier" not in df.columns:
        sys.exit("ERROR: combined_risk_tier not in prospective_final.csv")
    present_facts = [c for c in FACT_COLUMNS if c in df.columns]
    print("Model-vs-LLM tier classification (four-way)")
    print(f"  Provider/model : {args.provider} / {args.model}")
    print(f"  Features shown : {present_facts}")

    work = df.reset_index(drop=True)
    if args.limit:
        work = work.head(args.limit)
    print(f"  To classify    : {len(work):,} companies")

    out_path = NLP_DIR / "model_vs_llm.csv"
    done, results = set(), []
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        results = prev.to_dict("records")
        done = set(prev["company_num"].apply(num_key))
        print(f"Resume: {len(done):,} already done")

    pending = work[~work["company_num"].apply(num_key).isin(done)].reset_index(drop=True)
    start, processed = time.time(), 0
    for i, row in pending.iterrows():
        prompt = RISK_PROMPT.format(facts=build_facts(row, present_facts))
        raw = call_openai(client, prompt, args.model)

        if raw == "__DAILY_CAP__":
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"\nDaily cap reached. Saved {len(results):,} rows. "
                  f"Re-run with --resume tomorrow to continue.")
            return
        if not raw:
            print(f"  Skipping {row['company_num']} (call failed after retries).")
            continue

        tier, reason = parse(raw)
        if tier is None:
            print(f"  Skipping {row['company_num']} (unparseable response).")
            continue

        results.append({
            "company_num": row["company_num"],
            "company_name": row["company_name"],
            "combined_risk_tier": row["combined_risk_tier"],
            "llm_tier": tier,
            "llm_reason": reason,
        })
        processed += 1
        if processed % 10 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)
            el = time.time() - start
            rate = processed / el if el else 0
            rem = (len(pending) - (i + 1)) / rate if rate else 0
            print(f"  [{i+1:5d}/{len(pending)}] {el/60:.1f}min, ~{rem/60:.1f}min remain")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nDONE. ({(time.time()-start)/60:.1f} min)  Wrote {out_path}")
    print("Now score it:  python nlp_09_model_vs_llm.py --score")


if __name__ == "__main__":
    main()
