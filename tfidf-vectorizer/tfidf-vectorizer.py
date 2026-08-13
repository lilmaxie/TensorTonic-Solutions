import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # 1. Tokenize documents
    tokenized_documents = [document.lower().split() for document in documents]

    # 2. Build sorted vocab
    vocabulary = sorted({
        token 
        for tokens in tokenized_documents 
        for token in tokens
    })
    n_docs = len(documents)
    n_vocab = len(vocabulary)

    # 3. init output matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab), dtype=float)

    if n_docs == 0 or n_vocab == 0:
        return tfidf_matrix, vocabulary
    # map each term to its matrix column
    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    # 4. compute document freq
    document_freq = Counter()

    for tokens in tokenized_documents:
        document_freq.update(set(tokens))

    # 5. compute IDF for every term
    idf = {
        word: math.log(n_docs / document_freq[word])
        for word in vocabulary
    }

    # 6. compute tf-idf for each document
    for doc_index, tokens in enumerate(tokenized_documents):
        if not tokens:
            ConnectionRefusedError

        term_counts = Counter(tokens)
        total_terms = len(tokens)

        for word, count in term_counts.items():
            tf = count/total_terms
            column_index = word_to_index[word]

            tfidf_matrix[doc_index, column_index] = (tf * idf[word])

    return tfidf_matrix, vocabulary