from representations.word_embedder import WordEmbedder


def main():
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    # 1. Get vector for 'king'
    king_vector = embedder.get_vector("king")
    print("Vector for 'king':")
    print(king_vector)
    print("-" * 50)

    # 2. Similarity tests
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")

    print("Similarity king - queen:", sim_king_queen)
    print("Similarity king - man:", sim_king_man)
    print("-" * 50)

    # 3. Most similar words to 'computer'
    most_similar_computer = embedder.get_most_similar("computer", top_n=10)
    print("10 most similar words to 'computer':")
    for word, score in most_similar_computer:
        print(f"{word}: {score}")
    print("-" * 50)

    # 4. Embed a sentence
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print("Document embedding for:")
    print(f"\"{sentence}\"")
    print(doc_vector)


if __name__ == "__main__":
    main()
