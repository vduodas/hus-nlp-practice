from ..external_imports import Vectorizer, CountVectorizer    
        
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        """
        vectorizer: instance of your TfidfVectorizer or CountVectorizer
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        # 1. Vectorize input
        X = self.vectorizer.fit_transform(texts)

        # 2. Initialize logistic regression model
        self._model = LogisticRegression(solver="liblinear")

        # 3. Train
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        # 1. Transform new texts
        X = self.vectorizer.transform(texts)

        # 2. Predict
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
