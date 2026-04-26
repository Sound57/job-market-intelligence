"""
skill_extraction_ml.py
----------------------
TF-IDF vectorisation + KMeans clustering for skill extraction.

ML Component Added: KMeans Clustering
--------------------------------------
Instead of treating all job descriptions as a flat bag-of-words,
we cluster them into K topic groups (e.g., "Data Engineering",
"Frontend Dev", "DevOps"). This lets the app show which skills
dominate each cluster — a far more insightful result than a
single flat skill ranking.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def get_tfidf_features(
    corpus: pd.Series,
    max_features: int = 1500,
    min_df: int = 2,
    max_df: float = 0.85,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit a TF-IDF vectoriser on the corpus.

    Args:
        corpus:       Iterable of pre-processed text strings.
        max_features: Vocabulary cap.
        min_df:       Ignore terms in fewer than this many docs (removes rare noise).
        max_df:       Ignore terms in more than this fraction of docs (removes stopword-like terms).

    Returns:
        (X_tfidf dense array, fitted TfidfVectorizer)

    Bugs fixed vs original:
    - Added min_df / max_df to prune noise & near-stop-words that TF-IDF misses.
    - Returns dense array so KMeans and downstream code don't need .toarray() everywhere.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),   # capture bigrams like "machine learning", "data science"
        sublinear_tf=True,    # apply log(1+tf) — reduces impact of very frequent terms
    )
    X_sparse = vectorizer.fit_transform(corpus)
    X_dense = X_sparse.toarray()   # (n_docs, n_features) float64
    logger.info("TF-IDF matrix: %s", X_dense.shape)
    return X_dense, vectorizer


def extract_top_keywords(
    vectorizer: TfidfVectorizer,
    X: np.ndarray,
    top_n: int = 10,
) -> List[List[str]]:
    """
    For each document, return the top-N highest-TF-IDF terms.

    Bugs fixed vs original:
    - Original called row.toarray() on an already-dense array → AttributeError at runtime.
    - Now works on the dense ndarray returned by get_tfidf_features().

    Args:
        vectorizer: Fitted TfidfVectorizer.
        X:          Dense TF-IDF matrix (n_docs × n_features).
        top_n:      Number of keywords per document.

    Returns:
        List of keyword lists, one per document.
    """
    feature_names = vectorizer.get_feature_names_out()
    # argsort ascending → take last top_n in reverse = top scores
    sorted_idx = np.argsort(X, axis=1)[:, ::-1][:, :top_n]
    return [[feature_names[i] for i in row] for row in sorted_idx]


def cluster_jobs(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Group job descriptions into clusters using KMeans.

    Why KMeans here?
    - TF-IDF vectors live in high-dimensional space where cosine similarity
      works well. Normalising to unit vectors before KMeans approximates
      spherical k-means (cosine) clustering without extra dependencies.
    - 5 clusters is a sensible default for a general job board; the Streamlit
      UI exposes a slider so users can adjust this interactively.

    Args:
        X:            Dense TF-IDF matrix (already L2-normalised inside this fn).
        n_clusters:   Number of job clusters.
        random_state: For reproducibility.

    Returns:
        Integer array of cluster labels, shape (n_docs,).
    """
    X_norm = normalize(X)   # L2 normalise → cosine-approximate KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_norm)
    logger.info("KMeans clustering: %d clusters across %d documents", n_clusters, len(labels))
    return labels


def get_cluster_top_terms(
    vectorizer: TfidfVectorizer,
    X: np.ndarray,
    labels: np.ndarray,
    top_n: int = 10,
) -> dict:
    """
    For each cluster, return the top-N terms by mean TF-IDF score.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        X:          Dense TF-IDF matrix.
        labels:     Cluster label array from cluster_jobs().
        top_n:      Number of terms to return per cluster.

    Returns:
        Dict mapping cluster_id → list of (term, mean_score) tuples.
    """
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        mean_scores = X[mask].mean(axis=0)           # mean TF-IDF across cluster docs
        top_idx = np.argsort(mean_scores)[::-1][:top_n]
        cluster_terms[int(cluster_id)] = [
            (feature_names[i], round(float(mean_scores[i]), 4))
            for i in top_idx
        ]
    return cluster_terms