"""
Base interfaces and abstract classes for NLP components
Defines contracts for tokenizers, vectorizers, and models
"""
from abc import abstractmethod, ABC
from typing import List


class BaseTokenizer(ABC):
    """Abstract base class for text tokenizers"""
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class Vectorizer(ABC):
    """Abstract base class for text vectorizers"""
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass
    
    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass
    
    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass


class BaseModel(ABC):
    """Abstract base class for ML models"""
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
