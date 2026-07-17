import pandas as pd
cid = 707700  # WAVE I FINANCE, the one we've been looking at
cols = ["director_change_count", "name_change_count", "ar_filed_count", "total_submissions"]

f = pd.read_csv(r"data\processed\prospective_final.csv", low_memory=False)
print("prospective_final.csv :", f.loc[f.company_num == cid, [c for c in cols if c in f.columns]].to_dict("records"))

try:
    s = pd.read_csv(r"outputs\nlp\prospective_spv_labelled.csv", low_memory=False)
    print("spv_labelled.csv      :", s.loc[s.company_num == cid, [c for c in cols if c in s.columns]].to_dict("records"))
except Exception as e:
    print("spv_labelled read error:", e)

l = pd.read_csv(r"outputs\nlp\llm_features.csv", low_memory=False)
print("llm narrative          :", l.loc[l.company_num == cid, "audit_narrative"].head(1).tolist())