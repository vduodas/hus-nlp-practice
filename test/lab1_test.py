"""
Lab 1: Text Preprocessing & Tokenization
Tests for SimpleTokenizer and RegexTokenizer
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.preprocessing import SimpleTokenizer, RegexTokenizer
from src.core import load_raw_text_data_from


def test_tokenizers():
    """Test both SimpleTokenizer and RegexTokenizer"""
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
    dataset_path = r"C:\Users\DoubleDD\HUS\NLP&DL\datasets\UD_English-EWT\en_ewt-ud-train.txt"
    try:
        raw_text = load_raw_text_data_from(dataset_path)
        # Take a small portion of the text for demonstration
        sample_text = raw_text[:500] # First 500 characters
        print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
        print(f"Original Sample: {sample_text[:100]}...")
        simple_tokens = simple_tokenizer.tokenize(sample_text)
        print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
        regex_tokens = regex_tokenizer.tokenize(sample_text)
        print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
    except FileNotFoundError:
        print(f"\n⚠️  Dataset not found at {dataset_path}")
        print("Skipping dataset test")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LAB 1: Text Preprocessing & Tokenization")
    print("=" * 70 + "\n")
    test_tokenizers()
    print("\n✅ Lab 1 tests completed!\n")
