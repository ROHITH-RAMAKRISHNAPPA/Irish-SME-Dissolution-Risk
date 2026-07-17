"""
Stage 2 NLP - Step 2: topic modelling (four-method comparison).

Four topic models, one per family, for a methodology-comparison baseline:
  1. BERTopic (sentence-transformers + UMAP + HDBSCAN + class-TF-IDF) - neural, primary
  2. LDA via gensim                                                   - probabilistic
  3. NMF on TF-IDF                                                    - matrix factorization
  4. KMeans on the embeddings                                        - centroid baseline

Also reports an Adjusted Rand Index agreement matrix between the four methods,
so Chapter 5 can state whether they cluster the cohort consistently.

Output: outputs/nlp/topics.csv          (per-company assignments, all 4 methods)
        outputs/nlp/topic_summary.csv   (per-topic keywords + count, BERTopic)
        outputs/nlp/topic_agreement.csv (ARI matrix between methods)
        outputs/nlp/embeddings.npy      (sentence-transformer matrix)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR

NLP_DIR = OUTPUTS_DIR / "nlp"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# NACE Rev.2 division (first two digits of the code) -> section letter (A-U).
# Used to set the default topic count k = number of sectors present in the
# cohort, since the structured corpus text encodes sector membership.
_NACE_RANGES = [
    (1, 3, 'A'), (5, 9, 'B'), (10, 33, 'C'), (35, 35, 'D'), (36, 39, 'E'),
    (41, 43, 'F'), (45, 47, 'G'), (49, 53, 'H'), (55, 56, 'I'), (58, 63, 'J'),
    (64, 66, 'K'), (68, 68, 'L'), (69, 75, 'M'), (77, 82, 'N'), (84, 84, 'O'),
    (85, 85, 'P'), (86, 88, 'Q'), (90, 93, 'R'), (94, 96, 'S'), (97, 98, 'T'),
    (99, 99, 'U'),
]


def nace_section(code) -> str:
    """Map a NACE Rev.2 code ('8299', '82', '82.99') to its section letter."""
    s = str(code).strip().replace('.', '').replace(' ', '')
    if len(s) < 2 or not s[:2].isdigit():
        return ""
    d = int(s[:2])
    for lo, hi, sec in _NACE_RANGES:
        if lo <= d <= hi:
            return sec
    return ""


def count_nace_sections(df) -> int:
    """Number of distinct NACE sections present in the corpus."""
    if "nace_v2_code" not in df.columns:
        return 0
    secs = {nace_section(c) for c in df["nace_v2_code"].dropna()}
    secs.discard("")
    return len(secs)


def run_bertopic(texts, n_topics):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    print(f"  Embedding {len(texts):,} texts with {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

    print(f"  Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedder,
        nr_topics=n_topics or "auto",
        min_topic_size=10,
        verbose=True,
    )
    topic_ids, _probs = topic_model.fit_transform(texts, embeddings)

    labels = {}
    for tid in set(topic_ids):
        if tid == -1:
            labels[tid] = "outlier"
            continue
        top_words = [w for w, _ in topic_model.get_topic(tid)[:5]]
        labels[tid] = " / ".join(top_words)

    return topic_ids, labels, topic_model.get_topic_info(), embeddings


def run_kmeans(embeddings, k):
    from sklearn.cluster import KMeans
    print(f"  Fitting KMeans k={k}...")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(embeddings).tolist()


def run_lda(texts, k):
    from gensim import corpora
    from gensim.models import LdaModel

    print(f"  Tokenising {len(texts):,} texts for LDA...")
    tokens = [re.findall(r"\b[a-z]{3,}\b", t.lower()) for t in texts]
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow = [dictionary.doc2bow(toks) for toks in tokens]

    print(f"  Fitting LDA k={k}...")
    lda = LdaModel(bow, num_topics=k, id2word=dictionary,
                   passes=10, random_state=42)

    topic_ids = []
    for doc in bow:
        if not doc:
            topic_ids.append(-1)
            continue
        topic_probs = lda.get_document_topics(doc, minimum_probability=0)
        topic_ids.append(max(topic_probs, key=lambda x: x[1])[0])

    labels = {tid: " / ".join(w for w, _ in lda.show_topic(tid, topn=5))
              for tid in range(k)}
    return topic_ids, labels


def run_nmf(texts, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    print(f"  Vectorising {len(texts):,} texts (TF-IDF) for NMF...")
    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          min_df=5, max_df=0.5)
    X = vec.fit_transform(texts)

    print(f"  Fitting NMF k={k}...")
    nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)        # document-topic weights
    H = nmf.components_             # topic-term weights

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    arg = W.argmax(axis=1)
    topic_ids = [(-1 if row_sums[i] == 0 else int(arg[i])) for i in range(len(texts))]

    terms = vec.get_feature_names_out()
    labels = {}
    for tid in range(k):
        top = H[tid].argsort()[::-1][:5]
        labels[tid] = " / ".join(terms[j] for j in top)
    return topic_ids, labels


def agreement_matrix(assignments: dict) -> pd.DataFrame:
    """Adjusted Rand Index between every pair of methods."""
    from sklearn.metrics import adjusted_rand_score
    names = list(assignments.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = round(adjusted_rand_score(assignments[a], assignments[b]), 3)
    return mat


def main():
    ap = argparse.ArgumentParser(description="Stage 2 topic modelling (4-way)")
    ap.add_argument("--n_topics", type=int, default=None,
                    help="Topic count for KMeans/LDA/NMF (BERTopic auto-detects). "
                         "Default: number of NACE sections present in the cohort.")
    ap.add_argument("--skip_lda", action="store_true", help="Skip the LDA baseline")
    args = ap.parse_args()

    corpus_path = NLP_DIR / "corpus.csv"
    if not corpus_path.exists():
        sys.exit(f"ERROR: corpus not found at {corpus_path}. Run nlp_01_corpus.py first.")

    df = pd.read_csv(corpus_path, low_memory=False)

    # Default k = NACE sections present, since the structured text encodes sector.
    n_sections = count_nace_sections(df)
    if args.n_topics is not None:
        n_topics = args.n_topics
        k_source = "set via --n_topics"
    elif n_sections >= 2:
        n_topics = n_sections
        k_source = f"NACE sections present in cohort ({n_sections})"
    else:
        n_topics = 10
        k_source = "fallback default (no NACE codes found)"

    print(f"Stage 2 topic modelling (4-way)")
    print(f"  Companies: {len(df):,}")
    print(f"  n_topics:  {n_topics}  ({k_source})")

    # Cluster on name-free behavioural text when available so topics reflect
    # filing-event patterns rather than proper nouns in company names.
    text_col = "behaviour_text" if "behaviour_text" in df.columns else "combined_text"
    print(f"  Clustering text column: {text_col}")
    texts = df[text_col].fillna("").tolist()

    print("\n[1/4] BERTopic (neural)")
    bert_topics, bert_labels, bert_info, embeddings = run_bertopic(texts, n_topics)

    print("\n[2/4] LDA (probabilistic)")
    if not args.skip_lda:
        lda_topics, lda_labels = run_lda(texts, n_topics)
    else:
        print("  skipped (--skip_lda)")
        lda_topics, lda_labels = [-1] * len(df), {}

    print("\n[3/4] NMF (matrix factorization)")
    nmf_topics, nmf_labels = run_nmf(texts, n_topics)

    print("\n[4/4] KMeans (centroid)")
    km_topics = run_kmeans(embeddings, n_topics)

    df_out = df[["company_num", "company_name", "combined_risk_tier", "risk_score"]].copy()
    df_out["bertopic_id"] = bert_topics
    df_out["bertopic_label"] = [bert_labels.get(t, "outlier") for t in bert_topics]
    df_out["lda_id"] = lda_topics
    df_out["lda_label"] = [lda_labels.get(t, "") for t in lda_topics]
    df_out["nmf_id"] = nmf_topics
    df_out["nmf_label"] = [nmf_labels.get(t, "") for t in nmf_topics]
    df_out["kmeans_id"] = km_topics

    out_path = NLP_DIR / "topics.csv"
    df_out.to_csv(out_path, index=False)

    summary_path = NLP_DIR / "topic_summary.csv"
    bert_info.to_csv(summary_path, index=False)

    # Method-agreement matrix (Adjusted Rand Index)
    assignments = {"BERTopic": bert_topics, "LDA": lda_topics,
                   "NMF": nmf_topics, "KMeans": km_topics}
    ari = agreement_matrix(assignments)
    ari_path = NLP_DIR / "topic_agreement.csv"
    ari.to_csv(ari_path)

    emb_path = NLP_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)

    print(f"\nDONE.")
    print(f"  BERTopic discovered {len(bert_labels)} topics:")
    for tid in sorted(bert_labels.keys()):
        count = int((df_out["bertopic_id"] == tid).sum())
        print(f"    {tid:3d}: {bert_labels[tid][:60]:60s} ({count:,} companies)")
    print(f"\n  Method agreement (Adjusted Rand Index):")
    print(ari.to_string())
    print(f"\n  Per-company:  {out_path}")
    print(f"  Topic info:   {summary_path}")
    print(f"  Agreement:    {ari_path}")
    print(f"  Embeddings:   {emb_path}")


if __name__ == "__main__":
    main()
