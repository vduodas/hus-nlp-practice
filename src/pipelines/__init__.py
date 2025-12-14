"""
End-to-end NLP pipelines combining multiple components
"""
from ..preprocessing import RegexTokenizer
from ..representations import CountVectorizer
from ..models import TextClassifier


def create_text_classification_pipeline():
    """
    Create a complete text classification pipeline
    
    Returns:
        TextClassifier: Configured text classifier with preprocessor and vectorizer
    """
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    classifier = TextClassifier(vectorizer=vectorizer)
    
    return classifier


__all__ = [
    "create_text_classification_pipeline",
]
