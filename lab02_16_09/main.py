from lab01_16_09.preprocessing.regex_tokenizer import RegexTokenizer
from lab02_16_09.representations.count_vectorizer import CountVectorizer
from lab01_16_09.core.dataset_loaders import load_raw_text_data_from

def test_vectorizer():
    # Instantiate RegexTokenizer
    tokenizer = RegexTokenizer()
    
    # Instantiate CountVectorizer with the tokenizer
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    
    # Define sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    # Fit and transform the corpus
    document_term_matrix = vectorizer.fit_transform(corpus)
    
    # Print the learned vocabulary
    print("Learned Vocabulary:", vectorizer._vocabulary)
    
    # Print the document-term matrix
    print("Document-Term Matrix:", document_term_matrix)
    
    # path to dataset
    dataset_path = "C:\\Users\\DoubleDD\\Downloads\\UD_English-EWT\\UD_English-EWT\\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data_from(dataset_path)
    # Take 3 small portion of the text for demonstration
    sample_texts = [raw_text[:500],
                   raw_text[500:1000],
                   raw_text[1000:1500]] 
    print("\n--- Vectorizing Sample Text from UD_English-EWT ---")
    
    for idx, sample_text in enumerate(sample_texts, start=1):
        print(f"[Document {idx}] First 500 characters: {sample_text}\n")
        
    doc_term_matrix_for_UD_Eng = vectorizer.fit_transform(sample_texts)
    
    # print learned vocab
    print("Learned Vocabulary:", vectorizer._vocabulary)
    
    # print the doc-term matrix
    print("Document-term matrix:", doc_term_matrix_for_UD_Eng)
    
if __name__ == "__main__":
    test_vectorizer()