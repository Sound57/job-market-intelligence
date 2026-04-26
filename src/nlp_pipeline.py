"""
nlp_pipeline.py
---------------
Text cleaning and NLP processing using spaCy.
Handles model loading errors gracefully.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Load spaCy model once at module level — not inside functions (avoids reloading on every call)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )
except ImportError:
    raise ImportError("spaCy is not installed. Run: pip install spacy")


def clean_text(text: str) -> str:
    """
    Lowercase and strip non-alphabetic characters from text.

    Args:
        text: Raw job description string.

    Returns:
        Cleaned lowercase string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # only keep lowercase letters + spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_text(text: str) -> str:
    """
    Lemmatize tokens, removing stopwords, punctuation, and short tokens.

    Args:
        text: Pre-cleaned text string.

    Returns:
        Space-joined lemmatized token string.
    """
    if not text:
        return ""
    doc = nlp(text)
    tokens: List[str] = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_) > 2          # drop single/double-char noise tokens
        and token.lemma_.isalpha()         # drop numeric leftovers
    ]
    return " ".join(tokens)