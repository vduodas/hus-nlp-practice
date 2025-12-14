"""
Tokenizer implementations for text preprocessing
Provides simple and regex-based tokenization
"""
from core.interfaces import BaseTokenizer
import string
import re


class SimpleTokenizer(BaseTokenizer):
    """
    Simple character-based tokenizer that splits on whitespace and punctuation
    """
    def __init__(self, pattern: str = r''):
        super().__init__()
        self.pattern = pattern
    
    def tokenize(self, text):
        """
        Tokenize text by splitting on whitespace and punctuation
        
        Args:
            text: Input text string
            
        Returns:
            list[str]: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Initialize result list
        tokens = []
        current_token = ""
        
        # Process each character
        for char in text:
            # If character is punctuation, add current token (if any) and punctuation
            if char in string.punctuation:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            # If character is whitespace, add current token (if any)
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            # Add character to current token
            else:
                current_token += char
                
        # Add final token if exists
        if current_token:
            tokens.append(current_token)
            
        return tokens


class RegexTokenizer(BaseTokenizer):
    """
    Regex-based tokenizer using pattern matching
    """
    def __init__(self, pattern: str = r'\w+|[^\w\s]'):
        super().__init__()
        self.pattern = pattern
        
    def tokenize(self, text):
        """
        Tokenize text using regex pattern
        
        Args:
            text: Input text string
            
        Returns:
            list[str]: List of tokens
        """
        # Convert to lowercase and use regex to split
        # \w+ matches one or more word characters
        # [^\w\s] matches single punctuation characters
        return [token.lower() for token in re.findall(self.pattern, text.lower())]
