# -------------------------------------------------
# NLP Assignment
# Bag of Words, TF-IDF and Word2Vec
# -------------------------------------------------

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# ------------------ DATA ------------------
documents = [
    "I love natural language processing",
    "Natural language processing is interesting",
    "I love machine learning"
]

print("Documents:")
for doc in documents:
    print("-", doc)

print("\n-----------------------------------------\n")

# ================= BAG OF WORDS =================

# 1. Count Occurrence
print("1. Bag of Words (Count Occurrence):")

count_vectorizer = CountVectorizer()    
count_matrix = count_vectorizer.fit_transform(documents)    

print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("Count Matrix:\n", count_matrix.toarray())     

print("\n-----------------------------------------\n")

# 2. Normalized Count Occurrence
print("2. Normalized Count Occurrence:")

normalized_matrix = count_matrix.toarray().astype(float)
row_sums = normalized_matrix.sum(axis=1)

for i in range(len(row_sums)):
    if row_sums[i] != 0:
        normalized_matrix[i] = normalized_matrix[i] / row_sums[i]

print("Normalized Count Matrix:\n", normalized_matrix)

print("\n-----------------------------------------\n")

# ================= TF-IDF =================

print("3. TF-IDF:")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

print("\n-----------------------------------------\n")

# ================= WORD2VEC =================

print("4. Word2Vec Embeddings:")

# Tokenize sentences
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4    
)

# Display vector for a word
word = "language"
print(f"Word Embedding for '{word}':")
print(w2v_model.wv[word])

# Similar words
print("\nSimilar words to 'language':")    
print(w2v_model.wv.most_similar(word))

# -------------------------------------------------
# END OF PROGRAM
# -------------------------------------------------
  
    