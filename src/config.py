"""
Central configuration for the Irish SME Dissolution Risk Model.

Defines all project paths, raw and processed file locations, temporal split
dates, modelling thresholds, and the 84-feature column list used by the
deployed model. All scripts and notebooks import constants from this file.
"""

from pathlib import Path

# Directory layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

RAW_DIR_CRO   = RAW_DIR / "01_CRO_Raw"
RAW_DIR_CSO   = RAW_DIR / "02_CS0_Enrichment"
RAW_DIR_NACE  = RAW_DIR / "03_NACE_Reference"
RAW_DIR_NEXIS = RAW_DIR / "04_Nexis"
RAW_DIR_FAME  = RAW_DIR / "05_FAME"
RAW_DIR_ORBIS = RAW_DIR / "05_ORBIS"

for _d in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# Resolves a BRA file by its numeric suffix, allowing for date-stamped variants in the filename
def _find_bra(n: str) -> Path:
    matches = list(RAW_DIR_CSO.glob(f"BRA{n}.*.csv"))
    if matches:
        return sorted(matches)[0]
    return RAW_DIR_CSO / f"BRA{n}.csv"


# Raw input files (all data sources the pipeline reads from)
RAW_FILES = {
    "company_records":      RAW_DIR_CRO / "Company_Records.csv",
    "fs_2022":              RAW_DIR_CRO / "Financial_Statements_2022.csv",
    "fs_2023":              RAW_DIR_CRO / "Financial_Statements_2023.csv",
    "fs_2024":              RAW_DIR_CRO / "Financial_Statements_2024.csv",
    "dissolutions":         RAW_DIR_CRO / "Dissolutions_since_april_2025.csv",
    "cro_submissions_summary": RAW_DIR_CRO / "cro_submissions_summary.csv",
    "bra11": _find_bra("11"), "bra12": _find_bra("12"), "bra13": _find_bra("13"),
    "bra14": _find_bra("14"), "bra15": _find_bra("15"), "bra18": _find_bra("18"),
    "bra19": _find_bra("19"), "bra20": _find_bra("20"), "bra30": _find_bra("30"),
    "bra31": _find_bra("31"), "bra32": _find_bra("32"), "bra33": _find_bra("33"),
    "bra34": _find_bra("34"), "bra35": _find_bra("35"), "bra36": _find_bra("36"),
    "nace_table":  RAW_DIR_NACE / "NACE2.1-NACE2_Table_V1.05_(2).xlsx",
    "nace_corres": RAW_DIR_NACE / "CorresTab_NACE_Rev.2-NACE_Rev.2.1-TypeCorres_V1.05_(1).xlsx",
    "nace_notes":  RAW_DIR_NACE / "NACE_Rev2.1_Structure_Explanatory_Notes_EN_(4).xlsx",
    "fame_export":    RAW_DIR_FAME / "FAME_companies.xlsx",
    "fame_directors": RAW_DIR_FAME / "FAME_directors.xlsx",
    "orbis_ownership_raw":  RAW_DIR_ORBIS / "Orbis_ownership_raw.xlsx",
    "orbis_financials_raw": RAW_DIR_ORBIS / "Orbis_financials_raw.xlsx",
    "orbis_operations_raw": RAW_DIR_ORBIS / "Orbis_operations_raw.xlsx",
    "nexis_dir": RAW_DIR_NEXIS,
}

# Processed intermediate files (extraction outputs and model-ready splits)
PROCESSED_FILES = {
    "fame_companies":    PROCESSED_DIR / "fame_companies.csv",
    "fame_directors":    PROCESSED_DIR / "fame_directors.csv",
    "orbis_ownership":   PROCESSED_DIR / "orbis_ownership.csv",
    "orbis_financials":  PROCESSED_DIR / "orbis_financials.csv",
    "orbis_operations":  PROCESSED_DIR / "orbis_operations.csv",
    "director_dissolution": PROCESSED_DIR / "director_dissolution.csv",
    "nexis_mentions":    PROCESSED_DIR / "nexis_mentions.csv",
    "cro_charges":       PROCESSED_DIR / "cro_charges.csv",
    "master":            PROCESSED_DIR / "master.csv",
    "train_set":         PROCESSED_DIR / "train_set.csv",
    "test_set":          PROCESSED_DIR / "test_set.csv",
    "prospective_set":   PROCESSED_DIR / "prospective_set.csv",
    "prospective_final": PROCESSED_DIR / "prospective_final.csv",
}

# Temporal split anchors (companies observed before each cutoff form that split)
TRAIN_CUTOFF_DATE = "2022-12-31"
TEST_CUTOFF_DATE  = "2023-12-31"
OBS_DATE_STR      = "2024-12-31"
PROXY_CUTOFF_DATE = "2022-12-31"

# Active company filter: exclude companies whose last annual return is older than this many days at obs date
MAX_DAYS_SINCE_AR_AT_OBS = 730

# Filing behaviour thresholds
LATE_FILER_THRESHOLD_DAYS = 270
STATUTORY_WINDOW_DAYS     = 274
MAX_FILING_DELAY_DAYS     = 1825
SHORT_PERIOD_THRESHOLD    = 180
SAME_ADDRESS_MIN_COUNT    = 2

# Financial distress thresholds (Orbis ratio cutoffs for binary distress flags)
INSOLVENCY_SOLVENCY_THRESHOLD = 8.0
ILLIQUID_CURRENT_RATIO        = 1.0
HIGH_GEARING_THRESHOLD        = 200.0
SLOW_CREDITOR_DAYS            = 90

# Model and resampling settings
RANDOM_STATE       = 42
OPTUNA_N_TRIALS    = 100
CONTAMINATION      = 0.05
TOP_N_FLAGGED      = 0.05
PERMUTATION_N      = 1000
BOOTSTRAP_N        = 1000
CALIBRATION_SPLIT  = 0.5
FUZZY_MATCH_THRESH = 88

# CRO status taxonomy used by NB02 to derive the dissolution label
ACTIVE_STATUSES = {"Normal", "Strike Off Listed", "Normal "}

DISSOLVED_STATUSES = {
    "Dissolved", "Dissolved (liquidation)", "Dissolved (bankruptcy)",
    "Dissolved (merger or take-over)", "Dissolved (demerger)",
    "In liquidation", "Bankruptcy", "Liquidation",
    "Ceased IRL", "Ceased", "Struck off",
}

DISTRESS_STATUSES = {
    "Liquidation", "Ceased IRL", "Ceased",
    "Dissolved PostMerger", "Administration (UK)",
    "In Liquidation", "Bankruptcy",
}

# Company type encoding used for the company_type_enc feature
COMPANY_TYPE_MAP = {
    "LIMITED": 0, "LTD": 0,
    "PUBLIC LIMITED COMPANY": 1, "PLC": 1,
    "UNLIMITED": 2, "UNLIMITED COMPANY": 2,
    "DESIGNATED ACTIVITY COMPANY": 3, "DAC": 3,
    "COMPANY LIMITED BY GUARANTEE": 4, "CLG": 4,
    "SOCIETE EUROPEENNE": 5, "SE": 5,
    "ICAV": 6, "INVESTMENT COMPANY": 7,
    "INDUSTRIAL AND PROVIDENT": 8,
}

# Output subdirectories (created if absent so notebooks can write without checking)
FIGURES_DIR    = OUTPUTS_DIR / "figures"
TABLES_DIR     = OUTPUTS_DIR / "tables"
NARRATIVES_DIR = OUTPUTS_DIR / "narratives"

for _d in [FIGURES_DIR, TABLES_DIR, NARRATIVES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# The 84 modelling features grouped by data source. Two director-recency features
# were deliberately excluded because no Irish data source links directors across
# companies with dated dissolutions at scale (documented in Chapter 7).
FEATURE_COLS = [

    # CRO Filing Behaviour (8)
    "n_filings_yr0",
    "filing_consistency",
    "avg_days_to_file",
    "max_days_to_file",
    "sector_relative_deviation",
    "late_filer_flag",
    "short_period_flag",
    "annual_submission_rate",

    # Company Profile (5)
    "company_age_years",
    "nace_enc",
    "sector_imputed",
    "county_enc",
    "company_type_enc",

    # Name Signals (3)
    "name_risk_score",
    "name_generic_token_count",
    "name_has_numbers",

    # FAME / Director Signals (6)
    "fame_days_since_accounts",
    "fame_covered",
    "director_dissolution_count",
    "director_max_dissolutions",
    "director_back_to_back_flag",
    "director_avg_portfolio_size",

    # Sector / County Context (4)
    "sector_death_birth_ratio",
    "county_enterprise_density",
    "age_vs_sector_median",
    "sector_late_filer_risk",

    # Network / Dissolution Proxy (7)
    "name_address_dissolution_count",
    "name_token_dissolution_count",
    "name_token_dissolution_rate",
    "same_address_dissolution_count",
    "same_address_risk_flag",
    "same_day_reg_count",
    "same_day_dissolution_count",

    # Orbis Ownership (5)
    "is_corporate_owned",
    "is_foreign_owned",
    "guo_worldcompliance",
    "guo_irish_company_count",
    "n_subsidiaries_ult",

    # Orbis Financial Ratios (11)
    "solvency_ratio",
    "current_ratio",
    "is_loss_making",
    "is_insolvent",
    "pl_declining",
    "sol_declining",
    "illiquid",
    "solvency_trend_3yr",
    "roaa",
    "consecutive_loss_years",
    "ebit_margin",

    # Orbis Operational (9)
    "revenue_declining",
    "revenue_declining_2yr",
    "is_operating_loss",
    "ebit_declining",
    "highly_geared",
    "has_long_term_debt",
    "slow_creditor_payment",
    "employees_declining",
    "revenue_cagr_3yr",

    # CSO BRA Extended (5)
    "sector_employer_share",
    "county_enterprise_trend",
    "sector_birth_acceleration",
    "sector_newborn_micro_share",
    "sector_avg_startup_turnover",

    # Nexis News (1)
    "has_negative_news_mention",

    # CRO Charges (5)
    "charge_count",
    "outstanding_charge_count",
    "satisfied_charge_count",
    "total_charge_events",
    "days_since_last_charge",

    # CRO Submissions Core (3)
    "director_change_count",
    "ar_late_count",
    "name_change_count",

    # CRO Extended (12)
    "has_f8_before_obs",
    "has_examinership_before_obs",
    "has_winding_up_before_obs",
    "director_resignation_count",
    "ar_filed_count",
    "days_since_last_ar_filing",
    "total_submissions",
    "submission_history_years",
    "office_change_count",
    "days_since_last_office_change",
    "days_since_last_name_change",
    "other_form_count",
]

# Isolation Forest feature subset (18 features) used by NB05 for behavioural anomaly detection
IF_FEATURES = [
    "avg_days_to_file", "max_days_to_file", "late_filer_flag",
    "filing_consistency", "annual_submission_rate", "company_age_years",
    "same_day_reg_count", "same_day_dissolution_count",
    "same_address_dissolution_count", "director_avg_portfolio_size",
    "director_dissolution_count", "guo_irish_company_count",
    "is_corporate_owned", "is_foreign_owned",
    "solvency_ratio", "is_insolvent",
    "charge_count", "ar_late_count",
]

# Zero-coverage columns dropped from the modelling frame before training
# These are populated for no companies in the active set and carry no signal
DROP_COLS = [
    "gearing_ratio",
    "credit_period",
    "operating_revenue",
]

# Count features that are heavily right-skewed; a log1p companion is generated
# for the scale-sensitive models (Isolation Forest, Cox), not for the tree model
LOG_FEATURES = [
    "ar_filed_count",
    "total_submissions",
    "director_change_count",
    "same_day_reg_count",
    "name_token_dissolution_count",
    "charge_count",
    "office_change_count",
    "other_form_count",
]

# Financial ratio features with partial Orbis coverage; a binary present-flag is
# generated for each so the model can separate a real value from an imputed one
COVERAGE_FLAG_COLS = [
    "solvency_ratio",
    "current_ratio",
    "roaa",
    "ebit_margin",
    "revenue_cagr_3yr",
    "solvency_trend_3yr",
]

# Standardise the Isolation Forest and Cox feature matrices; tree models do not
# require this but distance and hazard calculations are scale-sensitive
SCALE_STAGE2_INPUTS = True

# Hard-fail guardrails - any duplicate or count mismatch indicates a broken feature list
assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), (
    f"Duplicate features: {[f for f in FEATURE_COLS if FEATURE_COLS.count(f) > 1]}"
)
assert len(FEATURE_COLS) == 84, f"Expected 84 features, got {len(FEATURE_COLS)}"
