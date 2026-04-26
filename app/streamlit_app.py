"""
streamlit_app.py
----------------
Job Market Intelligence Dashboard.
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import get_top_skills_from_keywords, get_top_locations, get_top_roles, summarise_clusters
from data_preprocessing import load_data, preprocess_data
from nlp_pipeline import process_text
from skill_extraction_ml import (
    cluster_jobs,
    extract_top_keywords,
    get_cluster_top_terms,
    get_tfidf_features,
)

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.5px; }
    .metric-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 4px;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #00f5a0; }
    .metric-card .label { font-size: 0.75rem; opacity: 0.7; letter-spacing: 1px; }
    .stPlotlyChart { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "monster_jobs.csv")

# ─── Cached data loading ─────────────────────────────────────────────────────────
# @st.cache_data prevents re-running expensive operations on every UI interaction.
# Bug in original: no caching at all — the entire pipeline re-ran on every widget change.

@st.cache_data(show_spinner="Loading & cleaning data…")
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = preprocess_data(df)
    return df


@st.cache_data(show_spinner="Running NLP pipeline…")
def run_nlp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["processed"] = df["cleaned"].apply(process_text)
    return df


@st.cache_data(show_spinner="Vectorising with TF-IDF…")
def run_tfidf(processed_series: pd.Series):
    return get_tfidf_features(processed_series)


@st.cache_data(show_spinner="Clustering job descriptions…")
def run_clustering(_X, n_clusters: int):
    # Leading underscore on _X tells Streamlit not to hash the numpy array
    return cluster_jobs(_X, n_clusters=n_clusters)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=64)
    st.title("⚙️ Controls")

    st.markdown("### 🔢 Cluster Settings")
    n_clusters = st.slider("Number of Job Clusters", min_value=2, max_value=10, value=5)
    top_n_skills = st.slider("Top N Skills to Display", min_value=5, max_value=30, value=15)

    st.markdown("---")
    st.markdown("### 📁 Data")
    custom_path = st.text_input("CSV Path (optional)", placeholder="data/monster_jobs.csv")
    data_path = custom_path if custom_path else DATA_PATH

    st.markdown("---")
    st.caption("Built with spaCy · scikit-learn · Streamlit")

# ─── Main header ─────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Job Market Intelligence")
st.markdown("*ML-powered skill extraction, clustering, and market analysis*")
st.markdown("---")

# ─── Load pipeline ───────────────────────────────────────────────────────────────
try:
    df = load_and_preprocess(data_path)
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ Data problem: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.stop()

df = run_nlp(df)
X, vectorizer = run_tfidf(df["processed"])
labels = run_clustering(X, n_clusters)
df["cluster"] = labels

df["keywords"] = extract_top_keywords(vectorizer, X)
top_skills = get_top_skills_from_keywords(df["keywords"], top_n=top_n_skills)
cluster_terms = get_cluster_top_terms(vectorizer, X, labels)
cluster_summary = summarise_clusters(df, cluster_terms)

# ─── KPI row ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, len(df), "TOTAL JOBS"),
    (c2, df["job_title"].nunique() if "job_title" in df.columns else "—", "UNIQUE ROLES"),
    (c3, n_clusters, "JOB CLUSTERS"),
    (c4, top_n_skills, "SKILLS TRACKED"),
]
for col, value, label in kpis:
    col.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── Tab layout ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Skills", "🔬 Job Clusters", "🏙️ Roles & Locations", "📋 Raw Data"])

# Tab 1: Top Skills
with tab1:
    st.subheader("Most In-Demand Skills (TF-IDF + Frequency)")

    skills, counts = zip(*top_skills)
    fig = px.bar(
        x=list(counts),
        y=list(skills),
        orientation="h",
        color=list(counts),
        color_continuous_scale="tealgrn",
        labels={"x": "Document Frequency", "y": "Skill"},
        title=f"Top {top_n_skills} Skills Across All Job Listings",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Space Mono",
        height=500,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # Word cloud via frequency table
    st.markdown("#### Skill Frequency Table")
    skill_df = pd.DataFrame(top_skills, columns=["Skill", "Count"])
    skill_df["Share %"] = (skill_df["Count"] / skill_df["Count"].sum() * 100).round(1)
    st.dataframe(skill_df, use_container_width=True, hide_index=True)

# Tab 2: Clustering
with tab2:
    st.subheader("KMeans Job Clusters")
    st.markdown(
        "Each cluster represents a **group of similar job descriptions** discovered automatically by KMeans. "
        "Top terms reveal the dominant theme of each cluster."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("#### Cluster Sizes")
        fig_pie = px.pie(
            cluster_summary,
            names="cluster",
            values="size",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Top Terms per Cluster")
        for _, row in cluster_summary.iterrows():
            with st.expander(f"🗂️ Cluster {row['cluster']}  —  {row['size']} jobs"):
                terms = cluster_terms.get(int(row["cluster"]), [])
                term_df = pd.DataFrame(terms, columns=["Term", "Mean TF-IDF Score"])
                fig_bar = px.bar(
                    term_df,
                    x="Mean TF-IDF Score",
                    y="Term",
                    orientation="h",
                    color="Mean TF-IDF Score",
                    color_continuous_scale="teal",
                )
                fig_bar.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# Tab 3: Roles & Locations
with tab3:
    col_r, col_l = st.columns(2)

    if "job_title" in df.columns:
        with col_r:
            st.subheader("Top Job Roles")
            roles = get_top_roles(df)
            fig_roles = px.bar(
                x=roles.values, y=roles.index,
                orientation="h",
                color=roles.values,
                color_continuous_scale="blues",
                labels={"x": "Count", "y": "Role"},
            )
            fig_roles.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roles, use_container_width=True)
    else:
        with col_r:
            st.info("'job_title' column not found in dataset.")

    if "location" in df.columns:
        with col_l:
            st.subheader("Top Locations")
            locs = get_top_locations(df)
            fig_locs = px.bar(
                x=locs.values, y=locs.index,
                orientation="h",
                color=locs.values,
                color_continuous_scale="purples",
                labels={"x": "Count", "y": "Location"},
            )
            fig_locs.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_locs, use_container_width=True)
    else:
        with col_l:
            st.info("'location' column not found in dataset.")

# Tab 4: Raw data
with tab4:
    st.subheader("Raw Dataset Preview")
    cols_to_show = [c for c in ["job_title", "location", "job_description", "cluster"] if c in df.columns]
    st.dataframe(df[cols_to_show].head(200), use_container_width=True)