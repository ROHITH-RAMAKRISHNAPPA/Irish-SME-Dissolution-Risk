"""
CRO submissions collector (single-pass, 814K Irish companies).

Pulls full submission histories for every Irish company from the CRO
Open Services API, categorises each filing (charges, strike-off,
examinership, winding-up, director changes, annual returns, office
changes, name changes, other), and writes per-category CSVs plus a wide
one-row-per-company summary file. Uses only the 15 fields the API
actually returns per submission (confirmed by --inspect mode):

  acc_year_to_date, company_bus_ind, company_num, doc_id, doc_num,
  doc_type_desc, file_size_bytes, num_pages, scan_date, scanned,
  sub_effective_date, sub_num, sub_received_date, sub_status_desc,
  sub_type_desc

Usage:
  # Inspect real API first (10 seconds)
  python src/01_collect_cro_submissions_all.py --inspect --api-key YOUR_KEY

  # Full 814K companies, ~75 hours, 4 workers
  python src/01_collect_cro_submissions_all.py --all-companies --workers 4 --api-key YOUR_KEY

  # Test run
  python src/01_collect_cro_submissions_all.py --limit 500 --api-key YOUR_KEY

Outputs (data/raw/01_CRO_Raw/):
  cro_submissions_raw.jsonl   Complete raw backup: re-extract anytime
  cro_charges.csv
  cro_strikeoff.csv
  cro_examinership.csv
  cro_winding_up.csv
  cro_director_changes.csv
  cro_ar_filings.csv
  cro_office_changes.csv
  cro_name_changes.csv
  cro_other_forms.csv         All other form types (catches anything missed)
  cro_submissions_summary.csv One wide row per company, all features

Safe to Ctrl-C. Re-run the same command to continue from checkpoint.
"""

import sys, time, base64, json, os, argparse, signal, threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_FILES

# Credentials
CRO_EMAIL   = "ramakrir@tcd.ie"
CRO_API_KEY = "PASTE_YOUR_KEY_HERE"

# Defaults
BASE_URL        = "https://services.cro.ie/cws"
DEFAULT_RATE    = 3.0
DEFAULT_WORKERS = 4
TIMEOUT         = 20
MAX_RETRIES     = 3
CHECKPOINT      = 1000

# Confirmed API fields (verified by --inspect mode)
# These are the ONLY 15 fields the API returns per submission.
# Do not reference any other field names.
API_FIELDS = [
    "acc_year_to_date",    # accounts year end date
    "company_bus_ind",     # business indicator
    "company_num",         # company number
    "doc_id",              # document ID
    "doc_num",             # document number
    "doc_type_desc",       # document type description
    "file_size_bytes",     # file size
    "num_pages",           # number of pages
    "scan_date",           # scan date
    "scanned",             # whether scanned
    "sub_effective_date",  # effective date    <<< NOT sub_eff_date
    "sub_num",             # submission number
    "sub_received_date",   # received date
    "sub_status_desc",     # status (Registered/Rejected/etc.)
    "sub_type_desc",       # form type description  <<< key for form detection
]

# Output paths
RAW_CRO       = PROJECT_ROOT / "data" / "raw" / "01_CRO_Raw"
RAW_CRO.mkdir(parents=True, exist_ok=True)

OUT_RAW       = RAW_CRO / "cro_submissions_raw.jsonl"
DONE_FILE     = RAW_CRO / "_cro_submissions_done.txt"
OUT_CHARGES   = RAW_CRO / "cro_charges.csv"
OUT_STRIKEOFF = RAW_CRO / "cro_strikeoff.csv"
OUT_EXAMINER  = RAW_CRO / "cro_examinership.csv"
OUT_WINDINGUP = RAW_CRO / "cro_winding_up.csv"
OUT_DIRECTORS = RAW_CRO / "cro_director_changes.csv"
OUT_AR        = RAW_CRO / "cro_ar_filings.csv"
OUT_OFFICE    = RAW_CRO / "cro_office_changes.csv"
OUT_NAMES     = RAW_CRO / "cro_name_changes.csv"
OUT_OTHER     = RAW_CRO / "cro_other_forms.csv"
OUT_SUMMARY   = RAW_CRO / "cro_submissions_summary.csv"


# Token-bucket rate limiter for the API
class RateLimiter:
    def __init__(self, rate):
        self._rate   = rate
        self._lock   = threading.Lock()
        self._tokens = rate
        self._last   = time.time()

    def acquire(self):
        with self._lock:
            now = time.time()
            self._tokens = min(self._rate,
                               self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                self._lock.release()
                time.sleep(wait)
                self._lock.acquire()
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# Field accessors (use only confirmed field names)
def _fk(sub):
    """Form key: lowercase sub_type_desc."""
    return str(sub.get("sub_type_desc") or "").lower().strip()


def _date(sub, *fields):
    """Return first non-empty date from confirmed field names."""
    for f in fields:
        val = str(sub.get(f) or "").strip()[:10]
        if val and len(val) >= 8 and "-" in val:
            return val
    return None


def _status(sub):
    return str(sub.get("sub_status_desc") or "").strip()


def _acc_year(sub):
    v = str(sub.get("acc_year_to_date") or "").strip()[:10]
    return v if (v and len(v) >= 4) else None


def _pages(sub):
    try:
        return int(sub.get("num_pages") or 0)
    except (ValueError, TypeError):
        return 0


def _size(sub):
    try:
        return int(sub.get("file_size_bytes") or 0)
    except (ValueError, TypeError):
        return 0


def is_form(fk, codes):
    """Match form key against codes using multiple patterns."""
    for c in [x.lower() for x in codes]:
        if fk == c:                   return True
        if fk.startswith(c + " "):    return True
        if fk.startswith(c + "-"):    return True
        if "form " + c in fk:         return True
        if " " + c + " " in fk:       return True
        if fk.endswith(" " + c):      return True
    return False


# Categorise a submission based on its form description.
# Returns one of: charge, strikeoff, examiner, windingup, director, ar, office, name, other
def categorise(fk):
    # Charges: C1, C1A, C1B (registration) and C6, C7, C17 (satisfaction)
    if (is_form(fk, ["c1","c1a","c1b","c1c","c2"])
            or is_form(fk, ["c6","c7","c17"])
            or "particulars of a charge" in fk
            or "particulars of charge" in fk
            or "satisfaction" in fk and "charge" in fk
            or "registration of a charge" in fk
            or "registration of charge" in fk
            or "floating charge" in fk):
        return "charge"

    # Strike-off: F8, F15 and related
    if (is_form(fk, ["f8","f15","f13","f16"])
            or "strike-off" in fk
            or "strike off" in fk
            or "struck off" in fk
            or "voluntary strike" in fk):
        return "strikeoff"

    # Examinership: E1, E2, E3
    if (is_form(fk, ["e1","e2","e3"])
            or "examin" in fk):
        return "examiner"

    # Winding up: D1, D2, D3, D4
    if (is_form(fk, ["d1","d2","d3","d4"])
            or "winding up" in fk or "winding-up" in fk
            or "liquidat" in fk
            or "receivership" in fk
            or "creditors" in fk and ("voluntar" in fk or "winding" in fk)):
        return "windingup"

    # Director/secretary changes: B10, H1, H2, H5, H5A
    if (is_form(fk, ["b10","h1","h2","h5","h5a","h15"])
            or "change director" in fk
            or "director or secretary" in fk
            or "change of director" in fk
            or "director details" in fk
            or "secretary details" in fk
            or "appointment of director" in fk
            or "resignation of director" in fk
            or "cessation of director" in fk
            or "new director" in fk):
        return "director"

    # Annual returns: B1, B1C, B1E, B10 (some B10 are annual return variants)
    if (is_form(fk, ["b1","b1c","b1e"])
            or "annual return" in fk
            or "b1 annual" in fk):
        return "ar"

    # Registered office changes: B2, B2A
    if (is_form(fk, ["b2","b2a"])
            or "registered office" in fk
            or "change of registered" in fk):
        return "office"

    # Name changes: G1, G1A (also catches "special resolution" which often accompanies name changes)
    if (is_form(fk, ["g1","g1a"])
            or "change of name" in fk
            or "change name" in fk):
        return "name"

    # Everything else logged for future use
    return "other"


def parse_all(company_num, subs):
    """
    Walk all submissions and bucket into categories.
    Only uses the 15 confirmed API fields.
    """
    if not isinstance(subs, list):
        subs = []

    charges=[]; strikeoff=[]; examiners=[]; windingup=[]
    directors=[]; ar_rows=[]; offices=[]; names=[]; other=[]

    for sub in subs:
        fk  = _fk(sub)
        dt  = _date(sub, "sub_received_date", "sub_effective_date")
        eff = _date(sub, "sub_effective_date")
        acc = _acc_year(sub)
        st  = _status(sub)
        pg  = _pages(sub)
        sz  = _size(sub)
        cat = categorise(fk)

        rec = {
            "date":             dt,
            "effective_date":   eff,
            "acc_year_to_date": acc,
            "status":           st,
            "form_desc":        fk,
            "num_pages":        pg,
            "file_size_bytes":  sz,
            "is_rejected":      int("reject" in st.lower()),
        }

        if cat == "charge":
            rec["is_floating"]  = int(
                "floating" in fk or "debenture" in fk or
                "all assets" in fk or "all present" in fk)
            rec["is_satisfied"] = int(
                is_form(fk, ["c6","c7","c17"]) or
                "satisfaction" in fk or "release" in fk)
            charges.append(rec)

        elif cat == "strikeoff":
            strikeoff.append(rec)

        elif cat == "examiner":
            examiners.append(rec)

        elif cat == "windingup":
            wtype = ("voluntary"            if "voluntar" in fk and "creditor" not in fk
                     else "creditors_voluntary" if "creditor" in fk
                     else "court"           if "court" in fk or is_form(fk,["d3"])
                     else "receivership"    if "receiver" in fk
                     else "other")
            rec["winding_type"] = wtype
            windingup.append(rec)

        elif cat == "director":
            # B10 / H5 etc.: can only count changes, not distinguish appointment vs resignation
            # UNLESS the form description explicitly says
            if any(w in fk for w in ("resign","cessation","terminat")):
                change_type = "resignation"
            elif any(w in fk for w in ("appoint","new director","new secretary")):
                change_type = "appointment"
            else:
                change_type = "change"
            rec["change_type"] = change_type
            directors.append(rec)

        elif cat == "ar":
            is_late = ("late" in fk or "no accounts" in fk)
            rec["is_late"] = int(is_late)
            ar_rows.append(rec)

        elif cat == "office":
            offices.append(rec)

        elif cat == "name":
            names.append(rec)

        else:
            other.append({
                "date":     dt,
                "status":   st,
                "form_desc": fk[:200],
            })

    return {
        "charges":   charges,   "strikeoff": strikeoff,
        "examiners": examiners, "windingup": windingup,
        "directors": directors, "ar":        ar_rows,
        "offices":   offices,   "names":     names,
        "other":     other,
    }


# Aggregate parsed submissions into one row per category per company
def aggregate(company_num, parsed):
    rows = {}

    # Charges
    cs   = parsed["charges"]
    reg  = [c for c in cs if not c["is_satisfied"]]
    sat  = [c for c in cs if     c["is_satisfied"]]
    cd   = [c["date"] for c in cs if c["date"]]
    rows["charges"] = {
        "company_num":              company_num,
        "charge_count":             len(reg),
        "has_floating_charge":      int(any(c["is_floating"] for c in reg)),
        "outstanding_charge_count": max(0, len(reg) - len(sat)),
        "satisfied_charge_count":   len(sat),
        "total_charge_events":      len(cs),
        "latest_charge_date":       max(cd) if cd else None,
        "earliest_charge_date":     min(cd) if cd else None,
        "rejected_charge_count":    sum(c["is_rejected"] for c in cs),
    }

    # Strike-off
    sf   = parsed["strikeoff"]
    sfd  = [s["date"] for s in sf if s["date"]]
    rows["strikeoff"] = {
        "company_num":    company_num,
        "f8_count":       len(sf),
        "has_f8_notice":  int(len(sf) > 0),
        "first_f8_date":  min(sfd) if sfd else None,
        "latest_f8_date": max(sfd) if sfd else None,
    }

    # Examinership
    ex   = parsed["examiners"]
    exd  = [e["date"] for e in ex if e["date"]]
    rows["examinership"] = {
        "company_num":         company_num,
        "under_examinership":  int(len(ex) > 0),
        "examiner_count":      len(ex),
        "first_examiner_date": min(exd) if exd else None,
    }

    # Winding-up
    wu   = parsed["windingup"]
    wud  = [w["date"] for w in wu if w["date"]]
    wut  = [w["winding_type"] for w in wu]
    rows["windingup"] = {
        "company_num":               company_num,
        "has_winding_up":            int(len(wu) > 0),
        "has_voluntary_winding_up":  int(any(t in ("voluntary","creditors_voluntary")
                                              for t in wut)),
        "has_court_winding_up":      int(any(t == "court" for t in wut)),
        "has_receivership":          int(any(t == "receivership" for t in wut)),
        "winding_up_type":           wut[0] if wut else None,
        "first_winding_up_date":     min(wud) if wud else None,
    }

    # Directors
    dirs    = parsed["directors"]
    resigns = [d for d in dirs if d["change_type"] == "resignation"]
    appts   = [d for d in dirs if d["change_type"] == "appointment"]
    dd      = [d["date"] for d in dirs if d["date"]]
    rows["directors"] = {
        "company_num":                   company_num,
        "director_change_count":         len(dirs),
        "director_resignation_count":    len(resigns),
        "director_appointment_count":    len(appts),
        "director_net_change":           len(appts) - len(resigns),
        "latest_director_change_date":   max(dd) if dd else None,
        "earliest_director_change_date": min(dd) if dd else None,
        "has_director_resigned":         int(len(resigns) > 0),
        "has_recent_director_change":    int(any(
            d["date"] and d["date"] >= "2022-01-01"
            for d in dirs if d["date"])),
    }

    # Annual returns
    ar      = parsed["ar"]
    ard     = [a["date"] for a in ar if a["date"]]
    late_n  = sum(a["is_late"] for a in ar)
    # Most recent acc_year_to_date across all AR filings
    accs    = [(a["date"] or "", a["acc_year_to_date"])
               for a in ar if a["acc_year_to_date"]]
    latest_acc = max(accs, key=lambda x: x[0])[1] if accs else None
    rows["ar"] = {
        "company_num":              company_num,
        "ar_filed_count":           len(ar),
        "ar_late_count":            late_n,
        "ar_on_time_count":         len(ar) - late_n,
        "first_ar_date":            min(ard) if ard else None,
        "latest_ar_date":           max(ard) if ard else None,
        "latest_acc_year_to_date":  latest_acc,
        "rejected_ar_count":        sum(a["is_rejected"] for a in ar),
    }

    # Office changes
    off  = parsed["offices"]
    offd = [o["date"] for o in off if o["date"]]
    rows["offices"] = {
        "company_num":               company_num,
        "office_change_count":       len(off),
        "latest_office_change_date": max(offd) if offd else None,
    }

    # Name changes
    nm   = parsed["names"]
    nmd  = [n["date"] for n in nm if n["date"]]
    rows["names"] = {
        "company_num":             company_num,
        "name_change_count":       len(nm),
        "latest_name_change_date": max(nmd) if nmd else None,
    }

    # Other forms (catch-all: logged for future re-extraction)
    oth  = parsed["other"]
    rows["other"] = {
        "company_num":       company_num,
        "other_form_count":  len(oth),
    }

    # Total submission stats
    all_subs = sum(len(v) for v in parsed.values())
    all_dates = [
        rec["date"]
        for cat in parsed.values()
        for rec in cat
        if isinstance(rec, dict) and rec.get("date")
    ]
    all_rejected = sum(
        rec.get("is_rejected", 0)
        for cat in parsed.values()
        for rec in cat
        if isinstance(rec, dict)
    )
    rows["totals"] = {
        "company_num":            company_num,
        "total_submissions":      all_subs,
        "first_submission_date":  min(all_dates) if all_dates else None,
        "latest_submission_date": max(all_dates) if all_dates else None,
        "total_rejected_filings": all_rejected,
        "has_any_rejected":       int(all_rejected > 0),
    }

    return rows


def make_summary(company_num, all_rows):
    row = {"company_num": company_num}
    for cat_rows in all_rows.values():
        for k, v in cat_rows.items():
            if k != "company_num":
                row[k] = v
    return row


# CSV schemas
SCHEMAS = {
    "charges":   ["company_num","charge_count","has_floating_charge",
                  "outstanding_charge_count","satisfied_charge_count",
                  "total_charge_events","latest_charge_date",
                  "earliest_charge_date","rejected_charge_count"],
    "strikeoff": ["company_num","f8_count","has_f8_notice",
                  "first_f8_date","latest_f8_date"],
    "examinership":["company_num","under_examinership","examiner_count",
                   "first_examiner_date"],
    "windingup": ["company_num","has_winding_up","has_voluntary_winding_up",
                  "has_court_winding_up","has_receivership",
                  "winding_up_type","first_winding_up_date"],
    "directors": ["company_num","director_change_count",
                  "director_resignation_count","director_appointment_count",
                  "director_net_change","latest_director_change_date",
                  "earliest_director_change_date","has_director_resigned",
                  "has_recent_director_change"],
    "ar":        ["company_num","ar_filed_count","ar_late_count",
                  "ar_on_time_count","first_ar_date","latest_ar_date",
                  "latest_acc_year_to_date","rejected_ar_count"],
    "offices":   ["company_num","office_change_count",
                  "latest_office_change_date"],
    "names":     ["company_num","name_change_count","latest_name_change_date"],
    "other":     ["company_num","other_form_count"],
    "totals":    ["company_num","total_submissions","first_submission_date",
                  "latest_submission_date","total_rejected_filings",
                  "has_any_rejected"],
}
OUT_FILES = {
    "charges":    OUT_CHARGES,  "strikeoff":   OUT_STRIKEOFF,
    "examinership":OUT_EXAMINER,"windingup":   OUT_WINDINGUP,
    "directors":  OUT_DIRECTORS,"ar":          OUT_AR,
    "offices":    OUT_OFFICE,   "names":       OUT_NAMES,
    "other":      OUT_OTHER,    "totals":      RAW_CRO/"cro_totals.csv",
}


# Buffered writer for incremental flushes
class Buffers:
    def __init__(self):
        self._lock     = threading.Lock()
        self.cats      = {cat: [] for cat in SCHEMAS}
        self.summary   = []
        self.raw_lines = []
        self.done_nums = []

    def add(self, cnum, subs, agg_rows, summary_row):
        with self._lock:
            for cat in self.cats:
                self.cats[cat].append(agg_rows[cat])
            self.summary.append(summary_row)
            self.raw_lines.append(
                json.dumps({"company_num": cnum, "submissions": subs},
                           separators=(",",":")))
            self.done_nums.append(cnum)

    def size(self):
        with self._lock:
            return len(self.done_nums)

    def flush(self):
        with self._lock:
            for cat, rows in self.cats.items():
                if not rows: continue
                df  = pd.DataFrame(rows, columns=SCHEMAS[cat])
                p   = OUT_FILES[cat]
                hdr = not p.exists() or p.stat().st_size == 0
                df.to_csv(p, mode="a", header=hdr, index=False)
                rows.clear()
            if self.summary:
                df  = pd.DataFrame(self.summary)
                hdr = not OUT_SUMMARY.exists() or OUT_SUMMARY.stat().st_size == 0
                df.to_csv(OUT_SUMMARY, mode="a", header=hdr, index=False)
                self.summary.clear()
            if self.raw_lines:
                with open(OUT_RAW, "a", encoding="utf-8") as f:
                    f.write("\n".join(self.raw_lines) + "\n")
                self.raw_lines.clear()
            if self.done_nums:
                with open(DONE_FILE, "a") as f:
                    f.write("\n".join(self.done_nums) + "\n")
                self.done_nums.clear()


# API
def auth_header(email, key):
    creds = base64.b64encode(f"{email}:{key}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Accept": "application/json"}


def test_auth(headers):
    r = requests.get(f"{BASE_URL}/company/368047/C?format=json",
                     headers=headers, timeout=15)
    if r.status_code == 200:
        print(f"  Auth OK: {r.json().get('company_name','?')}")
        return True
    print(f"  Auth FAILED: {r.status_code}")
    return False


def fetch_raw(session, cnum, headers, rate_limiter):
    url = f"{BASE_URL}/company/{cnum}/C/submissions"
    rate_limiter.acquire()
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=headers,
                            params={"format": "json"}, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for k in ("submissions","data","results","items","filings"):
                        if k in data and isinstance(data[k], list):
                            return data[k]
                    vals = [v for v in data.values() if isinstance(v, list)]
                    return vals[0] if vals else []
                return []
            if r.status_code == 404:   return []
            if r.status_code == 429:   time.sleep(15*(attempt+1)); continue
            if r.status_code in (401, 403):
                print(f"\n  FATAL AUTH ERROR {r.status_code}")
                return None
            time.sleep(3*(attempt+1))
        except requests.exceptions.Timeout:         time.sleep(5)
        except requests.exceptions.ConnectionError: time.sleep(10)
        except Exception:                           time.sleep(3)
    return []


# Inspect mode: print all fields from real API responses
def inspect_mode(headers, n=10):
    print("=" * 72)
    print("INSPECT: printing ALL fields from real API responses")
    print("=" * 72)
    test_nums = ["368047","10000","500000","600000","700000",
                 "800000","200000","300000","400000","150000"]
    session    = requests.Session()
    all_fields = defaultdict(set)
    all_forms  = set()

    for cnum in test_nums[:n]:
        url = f"{BASE_URL}/company/{cnum}/C/submissions"
        try:
            r = session.get(url, headers=headers,
                            params={"format": "json"}, timeout=15)
            if r.status_code != 200:
                print(f"\n  Company {cnum}: HTTP {r.status_code}")
                continue
            subs = r.json()
            if not isinstance(subs, list):
                subs = []
            print(f"\n  Company {cnum}: {len(subs)} submissions")
            for sub in subs[:5]:
                fk = _fk(sub)
                all_forms.add(fk)
                print(f"    sub_type_desc: '{fk}'")
                print(f"    ALL FIELDS: {sorted(sub.keys())}")
                for k in sub.keys():
                    all_fields[k].add(fk[:50])
        except Exception as e:
            print(f"  Company {cnum}: {e}")

    print("\n" + "="*72)
    print("CONFIRMED FIELD NAMES:")
    for f in sorted(all_fields):
        print(f"  {f}")
    print(f"\nForm descriptions seen ({len(all_forms)}):")
    for f in sorted(all_forms):
        print(f"  {f}")


# Company list
def load_company_list(use_all):
    if use_all:
        path = RAW_CRO / "Company_Records.csv"
        if not path.exists():
            print(f"ERROR: {path} not found"); sys.exit(1)
        df   = pd.read_csv(path, dtype={"company_num": str},
                           usecols=["company_num"])
        nums = sorted(df["company_num"].dropna()
                      .str.strip().str.zfill(6).unique().tolist())
        print(f"  Source: Company_Records.csv  ({len(nums):,} companies)")
        return nums
    nums = set()
    for key in ("fame_companies", "orbis_ownership"):
        p = PROCESSED_FILES.get(key)
        if p and Path(p).exists():
            df = pd.read_csv(p, dtype={"company_num": str},
                             usecols=["company_num"])
            df["company_num"] = df["company_num"].str.strip().str.zfill(6)
            before = len(nums)
            nums.update(df["company_num"].dropna().tolist())
            print(f"  {key}: {len(df):,} (+{len(nums)-before:,} new)")
    result = sorted(nums)
    print(f"  Total unique (FAME+Orbis): {len(result):,}")
    return result


def load_done():
    if DONE_FILE.exists():
        with open(DONE_FILE) as f:
            done = set(l.strip() for l in f if l.strip())
        print(f"  Already done: {len(done):,}")
        return done
    return set()


# Worker
_STOP = threading.Event()

def process_one(cnum, session, headers, rate_limiter):
    if _STOP.is_set():
        return cnum, None, None, None
    subs    = fetch_raw(session, cnum, headers, rate_limiter)
    if subs is None:
        _STOP.set()
        return cnum, None, None, None
    parsed  = parse_all(cnum, subs)
    agg     = aggregate(cnum, parsed)
    summary = make_summary(cnum, agg)
    return cnum, subs, agg, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-companies",  action="store_true")
    parser.add_argument("--inspect",        action="store_true")
    parser.add_argument("--limit",          type=int,   default=None)
    parser.add_argument("--workers",        type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--rate-limit",     type=float, default=DEFAULT_RATE)
    parser.add_argument("--api-key",        default=None)
    args = parser.parse_args()

    api_key = args.api_key or CRO_API_KEY
    if api_key == "PASTE_YOUR_KEY_HERE":
        print("ERROR: pass --api-key YOUR_KEY or set CRO_API_KEY in script")
        sys.exit(1)

    headers = auth_header(CRO_EMAIL, api_key)
    if not test_auth(headers):
        sys.exit(1)

    if args.inspect:
        inspect_mode(headers)
        return

    print("=" * 72)
    print("CRO SUBMISSIONS: FULL EXTRACTION")
    print("=" * 72)
    mode = "ALL 814K" if args.all_companies else "FAME+Orbis ~70K"
    print(f"Mode: {mode}  |  Workers: {args.workers}  |  Rate: {args.rate_limit}/s")
    print(f"Output: {RAW_CRO}")
    print()

    company_nums = load_company_list(args.all_companies)
    if args.limit:
        company_nums = company_nums[:args.limit]
        print(f"  TEST MODE: {args.limit:,} companies")

    done      = load_done()
    remaining = [n for n in company_nums if n not in done]
    eta_h     = len(remaining) / args.rate_limit / 3600
    print(f"\n  Remaining: {len(remaining):,}  ETA: ~{eta_h:.1f} hours\n")

    rl      = RateLimiter(args.rate_limit)
    buffers = Buffers()
    t0      = time.time()
    done_n  = [0]

    def _sig(*_):
        print("\nStopping: flushing...")
        _STOP.set()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    session = requests.Session()
    session.headers.update({"User-Agent": "ISD-Dissertation/1.0"})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_one, c, session, headers, rl): c
                for c in remaining}
        for fut in as_completed(futs):
            if _STOP.is_set():
                break
            cnum, subs, agg, summary = fut.result()
            if subs is None:
                break
            buffers.add(cnum, subs, agg, summary)
            done_n[0] += 1

            if done_n[0] % 100 == 0:
                el   = time.time() - t0
                rate = done_n[0] / el
                eta  = (len(remaining) - done_n[0]) / max(rate,0.01) / 3600
                pct  = done_n[0] / len(remaining) * 100
                print(f"  [{pct:5.1f}%] {done_n[0]:>7,}/{len(remaining):,}"
                      f"  {rate:.1f}/s  ETA {eta:.1f}h", flush=True)

            if done_n[0] % CHECKPOINT == 0:
                buffers.flush()
                raw_mb = OUT_RAW.stat().st_size/1e6 if OUT_RAW.exists() else 0
                print(f"  Checkpoint {done_n[0]:,}  raw={raw_mb:.0f}MB", flush=True)

    buffers.flush()

    # Final report
    total_done = len(load_done())
    print(f"\nDone. Total processed: {total_done:,}\n")
    for p in ([OUT_RAW] + list(OUT_FILES.values()) + [OUT_SUMMARY]):
        if p.exists():
            rows = sum(1 for _ in open(p,"rb")) - (0 if p.suffix==".jsonl" else 1)
            mb   = p.stat().st_size/1e6
            print(f"  {p.name:<52} {rows:>10,} rows  {mb:5.0f} MB")

    if OUT_SUMMARY.exists():
        s = pd.read_csv(OUT_SUMMARY, dtype={"company_num": str}, nrows=100000)
        print("\n  Signal coverage:")
        for col, label in [
            ("has_f8_notice",            "F8 strike-off notice"),
            ("under_examinership",       "Under examinership"),
            ("has_voluntary_winding_up", "Voluntary winding-up"),
            ("has_director_resigned",    "Director resigned"),
            ("has_recent_director_change","Director changed since 2022"),
            ("has_any_rejected",         "Has any rejected filing"),
            ("office_change_count",      "Changed registered office"),
            ("name_change_count",        "Changed company name"),
        ]:
            if col in s.columns:
                n = (s[col] > 0).sum() if col.endswith("_count") else s[col].sum()
                print(f"    {label:<44} {n:>8,}  ({n/len(s)*100:.1f}%)")

    print(f"\n  Raw backup: {OUT_RAW}")
    print("  To add new features later: python src/extract_from_raw.py")


if __name__ == "__main__":
    main()
