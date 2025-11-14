import pickle
from pathlib import Path
import re

import numpy as np
from nltk import download as nltk_download
from nltk.sentiment import SentimentIntensityAnalyzer

from text_processing import preprocess_single_text


MODEL_PATH = Path("static/model/model.pickle")
VOCAB_PATH = Path("static/model/vocabulary.txt")

# Load model with error handling
try:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}. Please train the model first.")
    model = None


def _load_tokens(path: Path = VOCAB_PATH):
    try:
        with path.open("r", encoding="utf-8") as vocab_file:
            return [line.strip() for line in vocab_file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Vocabulary file not found at {path}. Please train the model first.")
        return []


tokens = _load_tokens()
token_to_index = {token: idx for idx, token in enumerate(tokens)}

# Initialize VADER sentiment analyzer
try:
    _sentiment_analyzer = SentimentIntensityAnalyzer()
except LookupError:
    nltk_download('vader_lexicon')
    _sentiment_analyzer = SentimentIntensityAnalyzer()


def preprocessing(text: str):
    """Preprocess text for sentiment analysis"""
    if not text or not text.strip():
        return ""
    return preprocess_single_text(text)


def vectorizer(ds):
    """Convert preprocessed text to feature vectors"""
    if not tokens:
        return np.array([])
    
    # Handle single string input
    if isinstance(ds, str):
        ds = [ds]
    
    vocab_size = len(tokens)
    vectorized = np.zeros((len(ds), vocab_size), dtype=np.float32)
    
    for row_idx, sentence in enumerate(ds):
        if sentence:  # Check if sentence is not empty
            for word in sentence.split():
                idx = token_to_index.get(word)
                if idx is not None:
                    vectorized[row_idx, idx] = 1
    
    return vectorized


def get_prediction(vectorized_text):
    """Get sentiment prediction from the trained model"""
    if model is None:
        return None
    
    try:
        prediction = model.predict(vectorized_text)
        label = prediction[0] if hasattr(prediction, "__getitem__") else prediction
        # In training data: 0=positive, 1=negative
        return 'negative' if int(label) == 1 else 'positive'
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None


def enhanced_fallback_prediction(raw_text: str):
    """
    Enhanced fallback prediction using VADER with additional rules
    for better sentiment detection
    """
    if not raw_text or not raw_text.strip():
        return 'positive', 0.0
    
    # Get VADER scores
    polarity = _sentiment_analyzer.polarity_scores(raw_text)
    compound = polarity.get('compound', 0.0)
    
    # Enhanced rules for better classification
    text_lower = raw_text.lower()
    
    # Strong negative indicators
    strong_negative_words = [
        'terrible', 'awful', 'horrible', 'worst', 'hate', 'suck', 'bad', 
        'disappointing', 'useless', 'broken', 'failed', 'crash', 'slow',
        'annoying', 'frustrating', 'poor', 'cheap', 'waste', 'regret'
    ]
    
    # Strong positive indicators  
    strong_positive_words = [
        'amazing', 'excellent', 'fantastic', 'wonderful', 'perfect', 'love',
        'great', 'awesome', 'brilliant', 'outstanding', 'superb', 'incredible',
        'beautiful', 'happy', 'satisfied', 'impressed', 'recommend'
    ]
    
    # Count strong sentiment words
    negative_count = sum(1 for word in strong_negative_words if word in text_lower)
    positive_count = sum(1 for word in strong_positive_words if word in text_lower)
    
    # Adjust compound score based on strong indicators
    if negative_count > positive_count and negative_count > 0:
        compound = min(compound - 0.2, compound)  # Make more negative
    elif positive_count > negative_count and positive_count > 0:
        compound = max(compound + 0.2, compound)  # Make more positive
    
    # Classification with adjusted thresholds
    if compound <= -0.1:  # More sensitive negative threshold
        label = 'negative'
    elif compound >= 0.1:  # More sensitive positive threshold
        label = 'positive'
    else:
        # For neutral cases, lean slightly positive (common in product reviews)
        label = 'positive'
    
    return label, compound


def fallback_prediction(raw_text: str):
    """
    Use VADER sentiment as a safety net when the custom model
    cannot map the review onto the trained vocabulary.
    """
    return enhanced_fallback_prediction(raw_text)