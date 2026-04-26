# 🧠 Job Market Intelligence System

ML-powered skill extraction and job market analysis using spaCy, TF-IDF, and KMeans clustering.

## 📁 Folder Structure

```
job-market-intelligence/
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
├── src/
│   ├── nlp_pipeline.py       # spaCy text cleaning + lemmatisation
│   ├── data_preprocessing.py # CSV loading, validation, cleaning
│   ├── skill_extraction_ml.py# TF-IDF + KMeans clustering
│   └── analysis.py           # Aggregation, ranking, summaries
├── data/
│   └── monster_jobs.csv      # Input dataset
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## ⚡ Setup (from scratch)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Place your dataset
# Put monster_jobs.csv (must have a 'job_description' column) in data/

# 5a. Run Streamlit dashboard
streamlit run app/streamlit_app.py

# 5b. OR run CLI pipeline
python main.py
```

## 🧠 ML Components

| Component | What it does |
|---|---|
| TF-IDF (scikit-learn) | Scores terms by importance per document vs corpus |
| KMeans Clustering | Groups similar job descriptions into topic clusters |
| spaCy Lemmatisation | Reduces words to root form for cleaner vocabulary |

## 🔑 CSV Column Requirements

| Column | Required | Used for |
|---|---|---|
| `job_description` | ✅ Yes | NLP + skill extraction |
| `job_title` | Optional | Role analysis tab |
| `location` | Optional | Location analysis tab |
