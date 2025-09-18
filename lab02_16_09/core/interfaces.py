from abc import ABC, abstractmethod

class Vectorizer():
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass    
    
    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass
    
    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass