
"""
Support Ticket Intelligence Platform
Module: Text Preprocessing

Purpose:
    Clean and normalize support ticket descriptions
    before NLP feature extraction and model training.
"""

import re
import html
import unicodedata
from typing import Optional

import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 1. CONFIGURATION
# ============================================================

NEGATION_WORDS = {
    "not", "no", "nor", "never", "cannot"
}

# Preserve negation words when removing stopwords.
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS


CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are"
}


# ============================================================
# 2. BASIC TEXT VALIDATION
# ============================================================

def validate_text(text) -> str:
    """
    Handle missing values and convert input into string.
    """

    if text is None or pd.isna(text):
        return ""

    return str(text)


# ============================================================
# 3. UNICODE NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize different Unicode representations.
    """

    return unicodedata.normalize("NFKC", text)


# ============================================================
# 4. HTML REMOVAL
# ============================================================

def remove_html(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities.
    """

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    return text


# ============================================================
# 5. URL AND EMAIL REMOVAL
# ============================================================

def remove_urls(text: str) -> str:
    """
    Remove HTTP, HTTPS and www URLs.
    """

    return re.sub(
        r"(https?://\S+|www\.\S+)",
        " ",
        text,
        flags=re.IGNORECASE
    )


def remove_emails(text: str) -> str:
    """
    Remove email addresses.
    """

    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )


# ============================================================
# 6. LOWERCASE NORMALIZATION
# ============================================================

def convert_lowercase(text: str) -> str:
    """
    Convert text into lowercase.
    """

    return text.lower()


# ============================================================
# 7. CONTRACTION EXPANSION
# ============================================================

def expand_contractions(text: str) -> str:
    """
    Expand common English contractions.
    """

    pattern = re.compile(
        r"\b(" + "|".join(
            re.escape(word)
            for word in sorted(CONTRACTIONS, key=len, reverse=True)
        ) + r")\b",
        flags=re.IGNORECASE
    )

    return pattern.sub(
        lambda match: CONTRACTIONS[match.group(0).lower()],
        text
    )


# ============================================================
# 8. SPECIAL CHARACTER AND PUNCTUATION HANDLING
# ============================================================

def clean_special_characters(text: str) -> str:
    """
    Replace unwanted characters with spaces.

    Preserve:
        - Letters
        - Numbers
        - Hyphens
        - Underscores

    This helps preserve technical terms such as:
        ERR-500
        API_V2
        HTTP-404
    """

    text = re.sub(
        r"[^a-zA-Z0-9\s_-]",
        " ",
        text
    )

    return text


# ============================================================
# 9. WHITESPACE NORMALIZATION
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Remove unnecessary spaces, tabs and newlines.
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# 10. TOKENIZATION
# ============================================================

def tokenize_text(text: str) -> list:
    """
    Split text into tokens.

    Preserve technical tokens such as:
        api
        404
        err-500
        api_v2
    """

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*",
        text.lower()
    )


# ============================================================
# 11. STOPWORD REMOVAL
# ============================================================

def remove_stopwords(tokens: list) -> list:
    """
    Remove common English stopwords.

    Negation words are preserved.
    """

    return [
        word
        for word in tokens
        if word not in STOPWORDS
    ]


# ============================================================
# 12. STEMMING
# ============================================================

def apply_stemming(tokens: list) -> list:
    """
    Apply Porter stemming.

    Requires:
        pip install nltk
    """

    try:
        from nltk.stem import PorterStemmer

    except ImportError as error:
        raise ImportError(
            "Install NLTK using: pip install nltk"
        ) from error

    stemmer = PorterStemmer()

    return [
        stemmer.stem(word)
        for word in tokens
    ]


# ============================================================
# 13. LEMMATIZATION
# ============================================================

def apply_lemmatization(
    tokens: list,
    model_name: str = "en_core_web_sm"
) -> list:
    """
    Apply spaCy lemmatization.

    Requires:
        pip install spacy
        python -m spacy download en_core_web_sm
    """

    try:
        import spacy

    except ImportError as error:
        raise ImportError(
            "Install spaCy using: pip install spacy"
        ) from error

    try:
        nlp = spacy.load(
            model_name,
            disable=["parser", "ner"]
        )

    except OSError as error:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            "Install it using: "
            "python -m spacy download en_core_web_sm"
        ) from error

    doc = nlp(" ".join(tokens))

    return [
        token.lemma_
        for token in doc
    ]


# ============================================================
# 14. COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_text(
    text,
    remove_stop_words: bool = False,
    stemming: bool = False,
    lemmatization: bool = False
) -> str:
    """
    Execute the complete text preprocessing pipeline.

    Stemming and lemmatization are optional.
    Only one of them can be enabled at a time.
    """

    if stemming and lemmatization:
        raise ValueError(
            "Choose either stemming or lemmatization, not both."
        )

    # Basic validation
    text = validate_text(text)

    if not text.strip():
        return ""

    # Normalization
    text = normalize_unicode(text)

    # Cleaning
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)

    # Lowercase
    text = convert_lowercase(text)

    # Expand contractions before punctuation cleaning
    text = expand_contractions(text)

    # Character cleaning
    text = clean_special_characters(text)

    # Whitespace normalization
    text = normalize_whitespace(text)

    # Tokenization
    tokens = tokenize_text(text)

    # Optional stopword removal
    if remove_stop_words:
        tokens = remove_stopwords(tokens)

    # Optional stemming
    if stemming:
        tokens = apply_stemming(tokens)

    # Optional lemmatization
    if lemmatization:
        tokens = apply_lemmatization(tokens)

    return " ".join(tokens)


# ============================================================
# 15. DATAFRAME PROCESSING
# ============================================================

def process_dataframe(
    df: pd.DataFrame,
    text_column: str = "issue_description",
    output_column: str = "cleaned_description",
    remove_stop_words: bool = False,
    stemming: bool = False,
    lemmatization: bool = False
) -> pd.DataFrame:
    """
    Apply text preprocessing to a DataFrame.

    Original text is retained.
    Cleaned text is stored in a separate column.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataset."
        )

    processed_df = df.copy()

    processed_df[output_column] = (
        processed_df[text_column].apply(
            lambda text: preprocess_text(
                text,
                remove_stop_words=remove_stop_words,
                stemming=stemming,
                lemmatization=lemmatization
            )
        )
    )

    return processed_df


# ============================================================
# 16. SCRIPT EXECUTION AND VALIDATION
# ============================================================

if __name__ == "__main__":

    # Sample descriptions
    samples = [
        "URGENT! Cannot LOGIN to my ACCOUNT.",
        "API error 404 while accessing https://example.com",
        "<p>Payment FAILED. Customer can't access invoice.</p>",
        "   Database   connection   is NOT working!!!   ",
        "Error ERR-500 occurred in API_V2."
    ]

    print("=" * 65)
    print("SUPPORT TICKET TEXT PREPROCESSING")
    print("=" * 65)

    for index, sample in enumerate(samples, start=1):

        cleaned = preprocess_text(sample)

        print(f"\nExample {index}")
        print("Original:", sample)
        print("Cleaned :", cleaned)

    print("\n" + "=" * 65)
    print("PREPROCESSING VALIDATION COMPLETED")
    print("=" * 65)
