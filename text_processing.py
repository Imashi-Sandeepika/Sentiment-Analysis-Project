"""
Utility functions for keeping text preprocessing consistent
across both model training and inference.
"""
from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from nltk.stem import PorterStemmer

_STOPWORDS_PATH = Path("static/model/corpora/stopwords/english")
_porter = PorterStemmer()


def _load_stopwords(path: Path = _STOPWORDS_PATH) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


STOPWORDS = _load_stopwords()


def remove_punctuations(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def _drop_urls(token: str) -> str:
    return re.sub(r"^https?:\/\/.*[\r\n]*", "", token, flags=re.MULTILINE)


def preprocess_texts(texts: Iterable[str]) -> pd.Series:
    """
    Apply the same preprocessing pipeline that is used during
    inference to every text entry in `texts`.
    Returns a pandas Series of cleaned strings.
    """
    series = pd.Series(list(texts), dtype="string")
    series = series.fillna("")
    series = series.apply(lambda x: " ".join(x.lower() for x in x.split()))
    series = series.apply(lambda x: " ".join(_drop_urls(token) for token in x.split()))
    series = series.apply(remove_punctuations)
    series = series.str.replace(r"\d+", "", regex=True)
    series = series.apply(lambda x: " ".join(token for token in x.split() if token not in STOPWORDS))
    series = series.apply(lambda x: " ".join(_porter.stem(token) for token in x.split()))
    return series


def preprocess_single_text(text: str) -> pd.Series:
    """
    Convenience helper for preprocessing a single string and returning
    a pandas Series to align with the expectations of the existing
    vectorizer implementation.
    """
    return preprocess_texts([text])

