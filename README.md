# 🧠 NLP Assignments Collection (NLTK, ML & Deep Learning)

This repository contains a comprehensive set of Natural Language Processing (NLP) assignments implemented using Python. The tasks cover fundamental to advanced NLP techniques including tokenization, text preprocessing, vectorization, embeddings, Named Entity Recognition (NER), semantic analysis, and machine translation.

---

## 📌 Objectives

* Understand and implement core NLP preprocessing techniques
* Apply statistical and machine learning approaches to text data
* Build real-world NLP applications like NER and Machine Translation
* Gain hands-on experience with libraries such as NLTK, Scikit-learn, and Gensim

---

## 🛠️ Technologies & Libraries Used

* Python 🐍
* NLTK
* Scikit-learn
* Gensim (Word2Vec)
* Pandas, NumPy
* SpaCy / Transformers (for NER & MT, optional)

---

## 📂 Project Structure

```
NLP-Assignments/
│
├── Assignment1_Tokenization_Stemming/
├── Assignment2_Vectorization_Word2Vec/
├── Assignment3_TextCleaning_TFIDF/
├── Assignment4_NER_System/
├── Assignment5_Tokenization_Repeat/
├── Assignment6_Vectorization_Repeat/
├── Assignment7_TextProcessing_Repeat/
├── Assignment8_NER_Repeat/
├── Assignment9_WordNet/
├── Assignment10_MachineTranslation/
│
├── data/
├── outputs/
└── README.md
```

---

## 📖 Assignment Details

### 🔹 Assignment 1 & 5: Tokenization, Stemming, Lemmatization

* Implemented multiple tokenization techniques:

  * Whitespace Tokenization
  * Punctuation-based Tokenization
  * Treebank Tokenizer
  * Tweet Tokenizer
  * Multi-Word Expression (MWE) Tokenizer
* Applied:

  * Porter Stemmer
  * Snowball Stemmer
* Lemmatization using WordNet Lemmatizer

---

### 🔹 Assignment 2 & 6: Text Vectorization & Embeddings

* Bag of Words (BoW):

  * Count Occurrence
  * Normalized Count
* TF-IDF Vectorization
* Word Embeddings:

  * Word2Vec using Gensim

---

### 🔹 Assignment 3 & 7: Text Preprocessing Pipeline

* Text Cleaning:

  * Lowercasing
  * Removing punctuation & special characters
* Stopword Removal
* Lemmatization
* Label Encoding (for classification tasks)
* TF-IDF feature representation
* Outputs saved as CSV files

---

### 🔹 Assignment 4 & 8: Named Entity Recognition (NER)

* Built an NER system for extracting:

  * Person Names
  * Locations
  * Organizations
* Used:

  * SpaCy / NLTK / Transformers (depending on implementation)
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

### 🔹 Assignment 9: WordNet Semantic Analysis

* Used WordNet from NLTK to extract:

  * Synonyms
  * Antonyms
  * Hypernyms
* Explored semantic relationships between words

---

### 🔹 Assignment 10: Machine Translation System

* Developed a system to translate text between:

  * English ↔ Indian Language (e.g., Hindi/Marathi)
* Approaches:

  * Rule-based / Statistical / Neural (optional)
* Tools:

  * Transformers / Seq2Seq models / APIs

---

## 📊 Outputs

* Processed datasets stored in `/outputs`
* Includes:

  * Tokenized text
  * Cleaned text
  * TF-IDF vectors
  * Model predictions
  * Evaluation reports

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/NLP-Assignments.git
cd NLP-Assignments
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run any assignment:

```
python assignment1.py
```

---

## 📌 Future Improvements

* Add deep learning-based NLP models (BERT, LSTM)
* Improve NER accuracy using fine-tuned transformers
* Deploy Machine Translation system as a web app
* Integrate real-time datasets (Twitter, News APIs)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

---

## 📜 License

This project is open-source and available public.

---

## 👨‍💻 Author

**Avishkar Chothwe**
Third-Year Engineering Student | Aspiring ML Engineer

---
