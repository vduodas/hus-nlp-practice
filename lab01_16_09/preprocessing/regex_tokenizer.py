# preprocessing/regex_tokenizer.py

from ..core.interfaces import BaseTokenizer # type: ignore
import re

class RegexTokenizer(BaseTokenizer):
    def __init__(self, pattern: str = r'\w+|[^\w\s]'):
        super().__init__()
        self.pattern = pattern
        
    def tokenize(self, text):
        # Convert to lowercase and use regex to split
        # \w+ matches one or more word characters
        # [^\w\s] matches single punctuation characters
        return [token.lower() for token in re.findall(self.pattern, text.lower())]