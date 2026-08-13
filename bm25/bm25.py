import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    n_docs = len(docs)

    if n_docs == 0:
        return np.array([], dtype=float)

    scores = np.zeros(n_docs, dtype=float)

    if not query_tokens:
        return scores

    # document len and avg len
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avg_doc_length = np.mean(doc_lengths)

    # all docs are empty
    if avg_doc_length == 0:
        return scores

    # term freqs for each doc
    term_frequencies = [Counter(doc) for doc in docs]

    # doc freq for each term
    document_frequency = Counter()

    for doc in docs:
        document_frequency.update(set(doc))

    # remove duplicate query terms while preserving order
    unique_query_terms = list(dict.fromkeys(query_tokens))

    for term in unique_query_terms:
        df = document_frequency.get(term, 0)

        # term never appears in the corpus
        if df == 0:
            continue

        idf = math.log((n_docs-df+0.5) / (df+0.5) + 1.0)

        for doc_index in range(n_docs):
            tf = term_frequencies[doc_index].get(term, 0)

            if tf == 0:
                continue

            length_normalization = (1.0 - b + b*doc_lengths[doc_index] / avg_doc_length)

            numerator = tf * (k1+1.0)
            denominator = (tf + k1*length_normalization)

            scores[doc_index] += (idf * numerator / denominator)

    return scores
    