"""
Dataset loading utilities for NLP tasks
"""


def load_raw_text_data_from(path: str = None):
    """
    Load raw text data from a file
    
    Args:
        path: Path to the text file
        
    Returns:
        str: Raw text content
    """
    if not path:
        raise ValueError("Need to provide a valid path to the input file")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return content
