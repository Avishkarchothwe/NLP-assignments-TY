"""
NLP Text Processing Assignment
================================
Steps performed:
1. Text Cleaning
2. Lemmatization (rule-based suffix stripping)
3. Stop Word Removal
4. Label Encoding
5. TF-IDF Representation
6. Save all outputs to CSV
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# 1.  Sample Dataset
# ─────────────────────────────────────────────
data = {
    "text": [
        "The cats are running quickly through the beautiful gardens!",
        "Dogs barked loudly at the strangers walking by the street.",
        "She was eating delicious apples and oranges from the market.",
        "He is playing football with his friends on the ground.",
        "The children were studying mathematics and sciences in school.",
        "I love cooking tasty foods for my family every evening.",
        "They are swimming in the rivers and lakes during summers.",
        "The engineers are designing innovative machines for industries.",
        "Students were writing essays about historical events in class.",
        "Birds are flying high above the mountains and valleys.",
    ],
    "label": [
        "nature", "animals", "food", "sports", "education",
        "food", "nature", "technology", "education", "nature"
    ]
}

df = pd.DataFrame(data)
print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df.to_string(index=False))


# ─────────────────────────────────────────────
# 2.  Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    - Convert to lowercase
    - Remove punctuation
    - Remove numbers
    - Strip extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)                        # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()              # remove extra spaces
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

print("\n" + "=" * 60)
print("AFTER TEXT CLEANING")
print("=" * 60)
print(df[["text", "cleaned_text"]].to_string(index=False))


# ─────────────────────────────────────────────
# 3.  Stop Word Removal
#     (Manual English stop-word list – no NLTK needed)
# ─────────────────────────────────────────────
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "his", "her", "their"
])

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(filtered)

df["no_stopwords"] = df["cleaned_text"].apply(remove_stopwords)

print("\n" + "=" * 60)
print("AFTER STOP WORD REMOVAL")
print("=" * 60)
print(df[["cleaned_text", "no_stopwords"]].to_string(index=False))


# ─────────────────────────────────────────────
# 4.  Lemmatization
#     Rule-based suffix stripping (no NLTK required)
# ─────────────────────────────────────────────
IRREGULAR = {
    "running": "run",   "eating": "eat",   "playing": "play",
    "studying": "study","cooking": "cook", "swimming": "swim",
    "designing": "design","writing": "write","flying": "fly",
    "barked": "bark",   "walking": "walk", "cats": "cat",
    "dogs": "dog",      "friends": "friend","children": "child",
    "foods": "food",    "apples": "apple", "oranges": "orange",
    "rivers": "river",  "lakes": "lake",   "summers": "summer",
    "machines": "machine","industries": "industry",
    "events": "event",  "essays": "essay", "birds": "bird",
    "mountains": "mountain","valleys": "valley","strangers": "stranger",
    "gardens": "garden","engineers": "engineer","students": "student",
    "sciences": "science","families": "family",
}

SUFFIXES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"),  ("ising", "ise"),
    ("izing", "ize"), ("ising", "is"), ("ational", "al"),
    ("ousness", "ous"), ("iveness", "ive"), ("fulness", "ful"),
    ("ical", "ic"), ("ness", ""), ("ment", ""), ("ful", ""),
    ("ous", ""), ("ive", ""), ("ing", ""), ("ies", "y"),
    ("sses", "ss"), ("ied", "y"), ("ed", ""), ("ly", ""),
    ("er", ""), ("est", ""), ("s", ""),
]

def lemmatize_word(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix, replacement in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)] + replacement
    return word

def lemmatize_text(text: str) -> str:
    tokens = text.split()
    return " ".join(lemmatize_word(t) for t in tokens)

df["lemmatized"] = df["no_stopwords"].apply(lemmatize_text)

print("\n" + "=" * 60)
print("AFTER LEMMATIZATION")
print("=" * 60)
print(df[["no_stopwords", "lemmatized"]].to_string(index=False))


# ─────────────────────────────────────────────
# 5.  Label Encoding
# ─────────────────────────────────────────────
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

print("\n" + "=" * 60)
print("LABEL ENCODING")
print("=" * 60)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Mapping:", mapping)
print(df[["label", "label_encoded"]].to_string(index=False))


# ─────────────────────────────────────────────
# 6.  TF-IDF Representation
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=30)
tfidf_matrix = vectorizer.fit_transform(df["lemmatized"])

feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)
tfidf_df.insert(0, "label", df["label"].values)
tfidf_df.insert(1, "label_encoded", df["label_encoded"].values)

print("\n" + "=" * 60)
print("TF-IDF MATRIX (top 10 features shown)")
print("=" * 60)
print(tfidf_df.iloc[:, :12].to_string(index=False))


# ─────────────────────────────────────────────
# 7.  Save Outputs
# ─────────────────────────────────────────────
output_dir = "/mnt/user-data/outputs"

# Full pipeline dataframe
pipeline_df = df[["text", "cleaned_text", "no_stopwords", "lemmatized", "label", "label_encoded"]]
pipeline_df.to_csv(f"{output_dir}/pipeline_steps.csv", index=False)

# TF-IDF matrix
tfidf_df.to_csv(f"{output_dir}/tfidf_representation.csv", index=False)

# Label encoder mapping
mapping_df = pd.DataFrame(list(mapping.items()), columns=["label", "encoded_value"])
mapping_df.to_csv(f"{output_dir}/label_encoding_map.csv", index=False)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"  pipeline_steps.csv        → all NLP pipeline stages")
print(f"  tfidf_representation.csv  → TF-IDF feature matrix")
print(f"  label_encoding_map.csv    → label ↔ encoded value mapping")
print("\nAssignment Complete!")
