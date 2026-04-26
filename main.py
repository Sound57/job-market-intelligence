"""
main.py
-------
CLI entry point. Run the full pipeline end-to-end and print results.
Usage: python main.py
"""

import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Add src/ to path so relative imports work from the project root
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_data, preprocess_data
from nlp_pipeline import process_text
from skill_extraction_ml import (
    cluster_jobs,
    extract_top_keywords,
    get_cluster_top_terms,
    get_tfidf_features,
)
from analysis import get_top_skills_from_keywords, summarise_clusters


def run_pipeline(data_path: str = "data/monster_jobs.csv", n_clusters: int = 5) -> None:
    logger.info("=== Job Market Intelligence Pipeline ===")

    # 1. Load & preprocess
    df = load_data(data_path)
    df = preprocess_data(df)

    # 2. NLP
    logger.info("Running NLP pipeline (spaCy lemmatisation)…")
    df["processed"] = df["cleaned"].apply(process_text)

    # 3. TF-IDF
    X, vectorizer = get_tfidf_features(df["processed"])

    # 4. Keyword extraction
    df["keywords"] = extract_top_keywords(vectorizer, X)

    # 5. Skill ranking
    top_skills = get_top_skills_from_keywords(df["keywords"], top_n=20)
    print("\n🔥 Top 20 In-Demand Skills (TF-IDF):")
    for skill, count in top_skills:
        print(f"  {skill:<30} {count}")

    # 6. Clustering
    logger.info("Clustering job descriptions into %d groups…", n_clusters)
    labels = cluster_jobs(X, n_clusters=n_clusters)
    df["cluster"] = labels

    cluster_terms = get_cluster_top_terms(vectorizer, X, labels)
    summary = summarise_clusters(df, cluster_terms)

    print("\n📊 Job Clusters:")
    print(summary.to_string(index=False))

    print("\n🏷️  Top Terms per Cluster:")
    for cid, terms in cluster_terms.items():
        top = ", ".join(t for t, _ in terms[:8])
        print(f"  Cluster {cid}: {top}")


if __name__ == "__main__":
    run_pipeline()