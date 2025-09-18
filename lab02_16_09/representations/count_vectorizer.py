from lab02_16_09.core.interfaces import Vectorizer
from lab01_16_09.core.interfaces import BaseTokenizer
from lab01_16_09.preprocessing.simple_tokenizer import SimpleTokenizer


class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: BaseTokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SimpleTokenizer
        self._vocabulary: dict[str, int] = {}
        
    def fit(self, corpus: list[str]):
        # tokenize all of document within passed corpus (corpus is a list of documents)
        tokens_list = [self.tokenizer.tokenize(text=document) for document in corpus]   # -> list[list[token]]
        
        # flatten tokens_list and removed duplicates
        unique_tokens = set([token for tokens in tokens_list for token in tokens])
        
        # sorted version by alphabetical order
        unique_tokens_sorted = sorted(unique_tokens)
        
        # map each token in `unique_token_sorted` to its index -> return dict
        self._vocabulary = {token: idx for idx, token in enumerate(unique_tokens_sorted)}
        return True
        
    def transform(self, documents: list[str]) -> list[list[int]]:
        vectors = []
        vocab_size = len(self._vocabulary)
        
        for document in documents:
            vector = [0] * vocab_size
            tokens = self.tokenizer.tokenize(document)
            for token in tokens:
                if token in self._vocabulary:
                    vector[self._vocabulary[token]] += 1
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)
    