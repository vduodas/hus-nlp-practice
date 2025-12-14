"""
Lab 5 (Lab 4): Text Classification
Tests for TextClassifier with sentiment analysis
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split
from src.preprocessing import RegexTokenizer
from src.representations import CountVectorizer
from src.models import TextClassifier


def test_lab5_text_classifier():
    """Test TextClassifier with sentiment analysis data"""

    ## In-memory testing data
    # Dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0]

    # Split train/test (80:20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Instantiate tokenizer & vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)

    # Instantiate classifier
    classifier = TextClassifier(vectorizer)

    # Train
    classifier.fit(train_texts, train_labels)

    # Predict
    predictions = classifier.predict(test_texts)

    # Evaluate
    metrics = classifier.evaluate(test_labels, predictions)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("LAB 5: Text Classification & Sentiment Analysis")
    print("=" * 70 + "\n")
    test_lab5_text_classifier()
    print("\n✅ Lab 5 tests completed!\n")
