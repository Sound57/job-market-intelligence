"""
analysis.py
-----------
Aggregation and analytics on extracted keywords and clusters.
Pure functions — no side effects, no I/O.
"""

from collections import Counter
from typing import List, Tuple

import pandas as pd


def get_top_skills_from_keywords(
    keywords_list: List[List[str]],
    top_n: int = 20,
) -> List[Tuple[str, int]]:
    """
    Flatten all per-document keyword lists and count term frequency.

    Args:
        keywords_list: List of keyword lists (one per document).
        top_n:         How many top skills to return.

    Returns:
        List of (skill, count) tuples, sorted descending.
    """
    flat = [word for sublist in keywords_list for word in sublist if word]
    return Counter(flat).most_common(top_n)


def get_top_roles(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Return the top N most frequent job titles.

    Args:
        df:    DataFrame containing a 'job_title' column.
        top_n: Number of titles to return.

    Returns:
        pd.Series of value counts.

    Raises:
        KeyError: If 'job_title' column is absent.
    """
    if "job_title" not in df.columns:
        raise KeyError("DataFrame must contain a 'job_title' column.")
    return df["job_title"].value_counts().head(top_n)


def get_top_locations(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Return the top N most frequent job locations.

    Args:
        df:    DataFrame containing a 'location' column.
        top_n: Number of locations to return.

    Returns:
        pd.Series of value counts.

    Raises:
        KeyError: If 'location' column is absent.
    """
    if "location" not in df.columns:
        raise KeyError("DataFrame must contain a 'location' column.")
    return df["location"].value_counts().head(top_n)


def summarise_clusters(
    df: pd.DataFrame,
    cluster_top_terms: dict,
) -> pd.DataFrame:
    """
    Build a summary DataFrame showing cluster size and top 5 terms.

    Args:
        df:                DataFrame with a 'cluster' column.
        cluster_top_terms: Output of skill_extraction_ml.get_cluster_top_terms().

    Returns:
        DataFrame with columns: cluster, size, top_terms.
    """
    cluster_sizes = df["cluster"].value_counts().sort_index()
    rows = []
    for cluster_id, size in cluster_sizes.items():
        terms = cluster_top_terms.get(int(cluster_id), [])
        top_5 = ", ".join(t for t, _ in terms[:5])
        rows.append({"cluster": cluster_id, "size": size, "top_terms": top_5})
    return pd.DataFrame(rows)