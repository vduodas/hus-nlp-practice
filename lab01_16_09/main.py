# main.py
from .preprocessing.simple_tokenizer import SimpleTokenizer
from .preprocessing.regex_tokenizer import RegexTokenizer
from .core.dataset_loaders import load_raw_text_data_from

def test_tokenizers():
    # Test sentences
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    
    # Instantiate tokenizers
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    
    # Test both tokenizers
    print("Simple Tokenizer Results:")
    for sentence in sentences:
        tokens = simple_tokenizer.tokenize(sentence)
        print(f"Input: {sentence}")
        print(f"Tokens: {tokens}\n")
        
    print("Regex Tokenizer Results:")
    for sentence in sentences:
        tokens = regex_tokenizer.tokenize(sentence)
        print(f"Input: {sentence}")
        print(f"Tokens: {tokens}\n")
        
    # path to dataset
    dataset_path = "C:\\Users\\DoubleDD\\Downloads\\UD_English-EWT\\UD_English-EWT\\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data_from(dataset_path)
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500] # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")


if __name__ == "__main__":
    test_tokenizers()