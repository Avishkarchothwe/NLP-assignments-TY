# -------------------------------------------------
# Menu Driven NLP Program using NLTK
# Tokenization, Stemming and Lemmatization
# -------------------------------------------------

import nltk

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import (
    WhitespaceTokenizer,
    wordpunct_tokenize,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# ------------------ SAMPLE TEXT ------------------
text = "NLTK is amazing! I love NLP, machine learning, and AI. New York is a big city :) #NLP"

# ------------------ FUNCTIONS ------------------

def whitespace_tokenization():
    wt = WhitespaceTokenizer()
    print(wt.tokenize(text))

def punctuation_tokenization():
    print(wordpunct_tokenize(text))

def treebank_tokenization():
    tbt = TreebankWordTokenizer()
    print(tbt.tokenize(text))

def tweet_tokenization():
    tt = TweetTokenizer()
    print(tt.tokenize(text))

def mwe_tokenization():
    mwe = MWETokenizer([('New', 'York')], separator='_')
    print(mwe.tokenize(text.split()))

def porter_stemming():
    words = ["running", "runner", "ran", "easily", "fairly"]
    ps = PorterStemmer()
    for w in words:
        print(w, "->", ps.stem(w))

def snowball_stemming():
    words = ["running", "runner", "ran", "easily", "fairly"]
    ss = SnowballStemmer("english")
    for w in words:
        print(w, "->", ss.stem(w))

def lemmatization():
    words = ["running", "better", "cars", "went"]
    lemmatizer = WordNetLemmatizer()
    for w in words:
        print(w, "->", lemmatizer.lemmatize(w))

# ------------------ MENU ------------------

while True:
    print("\n================ NLP MENU =================")
    print("1. Whitespace Tokenization")
    print("2. Punctuation-based Tokenization")
    print("3. Treebank Tokenization")
    print("4. Tweet Tokenization")
    print("5. MWE Tokenization")
    print("6. Porter Stemmer")
    print("7. Snowball Stemmer")
    print("8. Lemmatization")
    print("9. Exit")
    print("==========================================")

    choice = input("Enter your choice (1-9): ")

    if choice == '1':
        whitespace_tokenization()
    elif choice == '2':
        punctuation_tokenization()
    elif choice == '3':
        treebank_tokenization()
    elif choice == '4':
        tweet_tokenization()
    elif choice == '5':
        mwe_tokenization()
    elif choice == '6':
        porter_stemming()
    elif choice == '7':
        snowball_stemming()
    elif choice == '8':
        lemmatization()
    elif choice == '9':
        print("Program Terminated.")
        break
    else:
        print("Invalid choice! Please enter valid option.")
