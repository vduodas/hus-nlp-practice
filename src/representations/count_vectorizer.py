"""
Count Vectorizer for converting text to count-based representations
"""
from core.interfaces import Vectorizer, BaseTokenizer
from preprocessing import SimpleTokenizer


class CountVectorizer(Vectorizer):
    """
    Converts a collection of text documents to a matrix of token counts
    """
    def __init__(self, tokenizer: BaseTokenizer = None):
        """
        Initialize CountVectorizer
        
        Args:
            tokenizer: Tokenizer instance. Defaults to SimpleTokenizer if None
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SimpleTokenizer()
        self._vocabulary: dict[str, int] = {}
        
    def fit(self, corpus: list[str]):
        """
        Learn vocabulary from corpus
        
        Args:
            corpus: List of text documents
            
        Returns:
            bool: True if fit successful
        """
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
        """
        Transform documents to count vectors
        
        Args:
            documents: List of text documents
            
        Returns:
            list[list[int]]: Document-term matrix (count vectors)
        """
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
        """
        Fit vocabulary and transform corpus in one step
        
        Args:
            corpus: List of text documents
            
        Returns:
            list[list[int]]: Document-term matrix
        """
        self.fit(corpus)
        return self.transform(corpus)
