# preprocessing/simple_tokenizer.py
from ..core.interfaces import BaseTokenizer
import string

class SimpleTokenizer(BaseTokenizer):
    def __init__(self, pattern: str = r''):
        super().__init__()
        self.pattern = pattern
    
    def tokenize(self, text):
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