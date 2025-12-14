import numpy as np
import gensim.downloader as api

from preprocessing.tokenizers import SimpleTokenizer

class WordEmbedder:
    def __init__(self, model_name: str):
        """
        Load word embedding model using gensim.downloader.
        Example model_name: 'glove-wiki-gigaword-50'
        """
        self.model_name = model_name
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.tokenizer = SimpleTokenizer()

    def get_vector(self, word: str):
        """
        Return embedding vector of a word.
        Handle OOV words by returning None.
        """
        word = word.lower()
        if word in self.model:
            return self.model[word]
        return None

    def get_similarity(self, word1: str, word2: str):
        """
        Return cosine similarity between two words.
        If either word is OOV, return None.
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.model or word2 not in self.model:
            return None

        return float(self.model.similarity(word1, word2))

    def get_most_similar(self, word: str, top_n: int = 10):
        """
        Return top N most similar words.
        If word is OOV, return empty list.
        """
        word = word.lower()
        if word not in self.model:
            return []

        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document: str):
        """
        Embed a document by averaging word vectors.
        - Tokenize document
        - Ignore OOV words
        - Return zero vector if no valid words
        """
        tokens = self.tokenizer.tokenize(document)

        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)
