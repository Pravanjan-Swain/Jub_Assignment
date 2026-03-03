
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents():
    documents = {
        "Leave Policy": open("leave_policy.txt", "r").read(),
        "IT Policy": open("it_policy.txt", "r").read(),
        "Travel Policy": open("travel_policy.txt", "r").read()
    }
    return documents


def prepare_data(documents):
    sentences = []
    sentence_sources = []

    for doc_name, content in documents.items():
        lines = content.split("\n")
        for line in lines:
            if line.strip() != "":
                sentences.append(line.strip())
                sentence_sources.append(doc_name)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    return sentences, sentence_sources, vectorizer, tfidf_matrix


def get_answer(query, sentences, sources, vectorizer, tfidf_matrix, threshold=0.5):

    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, tfidf_matrix)

    best_index = np.argmax(similarity)
    best_score = similarity[0][best_index]

    if best_score < threshold:
        return "Information not available in policy documents.", None

    return sentences[best_index], sources[best_index]