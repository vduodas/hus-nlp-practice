"""
Main src package - NLP library with modular components
"""
from .core import BaseTokenizer, Vectorizer, BaseModel, load_raw_text_data_from
from .preprocessing import SimpleTokenizer, RegexTokenizer
from .representations import CountVectorizer
from .models import TextClassifier
from .pipelines import create_text_classification_pipeline

__all__ = [
    # Core interfaces
    "BaseTokenizer",
    "Vectorizer",
    "BaseModel",
    "load_raw_text_data_from",
    # Preprocessing
    "SimpleTokenizer",
    "RegexTokenizer",
    # Representations
    "CountVectorizer",
    # Models
    "TextClassifier",
    # Pipelines
    "create_text_classification_pipeline",
]
