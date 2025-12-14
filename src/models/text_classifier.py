"""
Text Classification model combining vectorization and learning
"""
from typing import List, Dict
from core.interfaces import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    """
    Text classifier combining vectorization with logistic regression
    """
    def __init__(self, vectorizer: Vectorizer):
        """
        Initialize TextClassifier
        
        Args:
            vectorizer: Instance of a Vectorizer (e.g., CountVectorizer)
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the classifier on texts and labels
        
        Args:
            texts: List of text documents
            labels: List of class labels
        """
        # 1. Vectorize input
        X = self.vectorizer.fit_transform(texts)

        # 2. Initialize logistic regression model
        self._model = LogisticRegression(solver="liblinear")

        # 3. Train
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict class labels for new texts
        
        Args:
            texts: List of text documents
            
        Returns:
            List[int]: Predicted class labels
        """
        # 1. Transform new texts
        X = self.vectorizer.transform(texts)

        # 2. Predict
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Evaluate predictions with multiple metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict[str, float]: Dictionary with accuracy, precision, recall, f1 scores
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
