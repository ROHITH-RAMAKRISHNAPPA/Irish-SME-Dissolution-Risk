# Irish SME Dissolution Risk Model

An early-warning system that predicts Irish SME dissolution from public CRO filing-compliance metadata alone, with no financial-statement content, and triages 28,974 active companies into four operational risk tiers.


## Live Dashboard

**[Launch the Dissolution Risk Dashboard](https://your-app-name.streamlit.app)** - hosted on Streamlit Community Cloud


## Project Report & Data

**[Google Drive - Data](https://drive.google.com/drive/folders/1y0xqHsygUxtWCN4kH3yiEfyXiGky0Yy8?usp=drive_link)**

Includes the submitted PDF report, appendix, and the raw CRO, CSO, Orbis and Nexis files that are too large for GitHub.


## Overview

Corporate failure prediction has been built on financial ratios since Altman (1968). That approach carries an assumption the Irish register does not support: that the accounts exist. Filing of full financial statements is a minority behaviour across the 814,836 companies on the CRO register, so any ratio-based model is confined to the subset of firms that already disclose, which is not the subset most likely to fail.

This project inverts the question. Instead of asking what the accounts say, it asks whether the company files at all, how often, how late, and how that pattern changes. Filing-compliance metadata is universal, statutory, and free.

The result is that behaviour beats accounts. Removing all twenty financial features costs the model 3.5% of its Average Precision, while models built on financial ratios alone barely clear the base rate regardless of algorithm. What a company files carries the signal. What the filings say does not.


## Key Results

| Metric | Value |
|---|---|
| Register population | 814,836 companies |
| Modelled cohort | 222,321 (98,926 train / 94,421 test / 28,974 prospective) |
| Dissolution base rate | 6.69% train, 4.07% test |
| Winning model | XGBoost, Average Precision 0.6298, AUC-ROC 0.9412 |
| Class imbalance correction | scale_pos_weight = 13.94 |
| Financial ablation | AP 0.6333 to 0.6110 when all 20 financial features are dropped |
| Financials-only models | AP 0.048 to 0.060 across all five algorithms |
| Parsimony | top 15 of 84 features retain 89.3% of Average Precision |
| Triage yield | top 10% of the test population recovers 79.0% of dissolutions, 7.9x random |
| Lead time | median 14.6 months before statutory dissolution |
| Six-month detection | 40.2% of 3,678 companies flagged at that horizon, p < 1e-300 |
| Anomaly layer | Isolation Forest AUC 0.5297 against a permutation null of 0.5080, p < 0.0001 |


## Model Comparison

| Model | Average Precision | AUC-ROC | F1 | KS | Brier |
|---|---|---|---|---|---|
| **XGBoost** | **0.6298** | 0.9412 | 0.5933 | 0.7400 | 0.0272 |
| LightGBM | 0.6012 | 0.9313 | 0.5752 | 0.7148 | 0.0287 |
| Logistic Regression | 0.5353 | 0.9679 | 0.6299 | 0.8463 | 0.0532 |
| Random Forest | 0.4522 | 0.8810 | 0.4489 | 0.6074 | 0.0725 |
| Decision Tree | 0.1771 | 0.7846 | 0.2468 | 0.4562 | 0.2297 |

Logistic regression wins on AUC and loses on Average Precision. At a 4% base rate, AUC rewards ranking the negative majority correctly while Average Precision measures whether the top of the list is worth reviewing. The second question is the one an audit team asks, so XGBoost is selected.


## Architecture

**Stage 1** XGBoost produces a calibrated dissolution probability. Isotonic calibration is fitted on a held-out half of the temporal test set and improves the Brier score from 0.0272 to 0.0226.

**Stage 2** an Isolation Forest flags companies whose filing behaviour is anomalous, independently of the supervised score.

The two stages combine into four operational tiers. The gate is the top 20% by Stage 1 score, not the top 5%.

| Tier | Companies | Definition |
|---|---|---|
| PRIORITY | 78 | Stage 1 High (top 5%) AND anomaly-flagged. Both signals agree. |
| DISSOLUTION_RISK | 5,833 | Stage 1 High or Medium (top 20%), no anomaly flag. Holds 1,454 companies in the same top-5% band as PRIORITY. |
| BEHAVIOURAL_ANOMALY | 1,578 | Anomaly-flagged, Stage 1 Low. Unusual filing, unremarkable score. |
| LOW_CONCERN | 21,485 | Neither. |

The tiers are a reason code, not a work order. PRIORITY is an intersection of 78 companies out of the 1,532 in Stage 1 High, so the highest-scoring company in the cohort is likely to sit in DISSOLUTION_RISK.


## Language-Model Layer

A language model receives each company's top SHAP drivers and eight behavioural features, with no retrieval and no raw filings, and returns a plain-language summary and a set of distress signals. It cannot introduce a fact the classifier did not observe.

| Result | Value |
|---|---|
| Narratives generated | 28,974, zero blanks, 21,090 distinct distress-signal strings |
| Confabulation screen | 6,199 candidates raised, all confirmed as false positives |
| Cited features in the top five SHAP drivers | 72.3% |
| Narratives citing a top-three driver | 94.3% |
| Entity classification | 28,974 companies, 9,828 (34%) returned as uncertain |

A second pass classifies entity type. This matters more than it sounds: **76% of the PRIORITY tier are special-purpose vehicles**, wound up on schedule once their purpose is served rather than failing. Without that distinction the tier exit rates cannot be interpreted.

The concordance figures measure faithfulness, not discovery. The prompt supplies SHAP-relevant features, so a high score establishes that the text emphasises what actually moves the score. It does not establish that the model recovered those drivers independently.


## Repository Structure
```
├── Irish_SME_Dissolution_Risk.py   # Streamlit dashboard, 7 tabs
├── per_company_report.py           # per-company audit PDF
├── build_model_card.py             # governance model card PDF
├── build_helix_export.py           # flat-file export for an analytics platform
├── test_recalibration.py           # six-calibrator comparison, see Limitations
├── ireland_counties.geojson
│
├── notebooks/
│   ├── 00_config.ipynb                      # paths, constants, feature registry
│   ├── 01_data_loading.ipynb                # CRO, CSO, Orbis, Nexis ingestion
│   ├── 02_feature_engineering_rebuilt.ipynb # 84 features, train-only winsorisation
│   ├── 03_model_training.ipynb              # 5 models, Optuna, isotonic calibration
│   ├── 04_anomaly_detection.ipynb           # Isolation Forest, LOF, permutation nulls
│   ├── 05_shap_explainability.ipynb         # global and per-company SHAP
│   ├── 06_figures.ipynb                     # 23 dissertation figures, runs last
│   ├── ablation_financial_value.ipynb       # the thesis experiment
│   └── ablation_annual_submission_rate.ipynb
│
├── src/
│   ├── 01_collect_cro_submissions_all.py    # CRO API collection
│   ├── extract_from_raw.py
│   ├── extract_orbis.py / extract_fame.py / extract_nexis.py
│   ├── extract_director_dissolution.py
│   ├── tier_confirmation_rate.py            # register cross-check
│   └── config.py
│
├── nlp/
│   ├── nlp_01_corpus.py / nlp_02_sequence_enrich.py
│   ├── nlp_04_llm_extract.py                # 28,974 narratives
│   ├── nlp_05_validation.py                 # register cross-check
│   ├── nlp_06_entity_type.py                # SPV / holding / trading / uncertain
│   ├── nlp_07_spv_rules.py                  # rule vs model agreement
│   ├── nlp_08_shap_llm_concordance.py
│   └── nlp_09_model_vs_llm.py
│
├── models/          # winning model, isotonic calibrator, feature list, metadata
├── outputs/         # scored companies, 48 tables, 23 figures
├── data/processed/  # prospective_final.csv, the scored cohort
└── requirements.txt
```


## Run Order

```bash
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute notebooks/00_config.ipynb
jupyter nbconvert --to notebook --execute notebooks/01_data_loading.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering_rebuilt.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_anomaly_detection.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_shap_explainability.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_figures.ipynb

python nlp/nlp_04_llm_extract.py
python nlp/nlp_06_entity_type.py
python nlp/nlp_05_validation.py
python nlp/nlp_07_spv_rules.py
python nlp/nlp_08_shap_llm_concordance.py

streamlit run Irish_SME_Dissolution_Risk.py
```

nlp_05 and nlp_07 are re-run after nlp_06 because both consume its output.

06_figures runs last because it reads the trained model, the model comparison
table, and the Stage 2 outputs. It builds the dissertation figures rather than
exploring the data, so it cannot run before the results it draws.


## Data Sources

- **CRO filing metadata**: [CRO open data](https://opendata.cro.ie), collected by `src/01_collect_cro_submissions_all.py` with a free API key. Company records, submission history, annual returns, director changes, office changes, charges, strike-offs and windings-up.
- **CSO enrichment**: [Central Statistics Office](https://www.cso.ie) business demography, sector survival and birth rates.
- **Orbis and FAME**: financial statements, accessed under a subscription licensed to Trinity College Dublin. Used only in the ablation study that shows they are not needed. Not redistributable.
- **LexisNexis**: news mentions, academic licence.

Raw CRO, CSO, Orbis and Nexis files are not tracked in this repository due to file size. `cro_submissions_raw.jsonl` alone is 8 GB, and two further files exceed GitHub's 100 MB limit. The training and test partitions (86 MB and 83 MB) are also excluded; the notebooks rebuild them from raw in order. See the Google Drive folder above.


## Temporal Design

Every company is scored at an observation date, and its outcome is read from a 24-month window that begins after that date. Winsorisation bounds and all scaling are fitted on the training fold and applied to the others, so no test or prospective information reaches training. The prospective partition has an observation date of 31 December 2024 and no outcome label; it is what the dashboard shows.


## Limitations

**The calibrated probability saturates at both ends.** Isotonic calibration is bounded by the dissolution rate observed within each score band, so 25 of the 28,974 companies return exactly 1.0 and 1,517 return 0.0. These are the endpoints of the calibration, not statements of certainty. `test_recalibration.py` compares six calibrators and shows why the current one was kept: every alternative that removes the ceiling makes the Brier score worse, and at the prior weight needed to clear the floor, 1,797 companies change risk band.

**The model ranks, it does not diagnose.** A high score means a filing pattern resembling companies that dissolved. It does not establish cause and it is not a determination of viability.

**Absence of signal is not clearance.** A company at the floor is one the model has nothing to say about, which is not the same as a company that is safe.

**Scores describe each company at its observation date**, not today. A company that has filed, been struck off, or changed hands since will not reflect that.

**Special-purpose vehicles dominate the top tier.** Their dissolution is a scheduled deal-end. The entity classifier separates them, but with 34% returned as uncertain where the company name carries no signal.


## Team

- **Rohith Ramakrishnappa**
- **Lexi Miller**
- **Raahem Ahmed**

Supervisor: Prof. Baidyanath Biswas, Trinity College Dublin.

Industry partner: Hannah Beckett, Manager, Technology Consulting, EY Ireland.


## References

- Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4).
- Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD 2016*.
- du Jardin, P. (2021). Forecasting corporate failure using ensemble of self-organizing neural networks. *European Journal of Operational Research*, 288(3).
- Liu, F.T., Ting, K.M. and Zhou, Z.-H. (2008). Isolation Forest. *ICDM 2008*.
- Lundberg, S.M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
- Zadrozny, B. and Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD 2002*.
