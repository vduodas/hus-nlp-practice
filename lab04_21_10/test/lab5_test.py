from sklearn.model_selection import train_test_split
from lab04_21_10.external_imports import RegexTokenizer, CountVectorizer
from lab04_21_10.models.text_classifier import TextClassifier

def test_lab5_text_classifier():

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
    test_lab5_text_classifier()
