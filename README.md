## Multi-Label Text Classification using Word2Vec, GloVe, FastText & BERT (PyTorch)
This project provides a complete end-to-end workflow for multi-label text classification, comparing four embedding techniques:

Word2Vec

GloVe

FastText

BERT (Sentence-Transformers)

Using the Consumer Review of Clothing Products dataset from Kaggle, the project demonstrates how contextual embeddings significantly outperform traditional static embeddings in modern NLP tasks.

## 📌 Highlights
Multi-label text classification

Identical evaluation setup across four embedding models

Clean preprocessing and embedding generation pipeline

PyTorch-based deep learning classifier

Automatic export of evaluation metrics and visualization plots

Embedding caching to speed up repeated experiments

Single-script reproducible workflow

## 📂 Dataset
Property	Details
Name	Consumer Review of Clothing Product
Source	Kaggle
Dataset Link	https://www.kaggle.com/datasets/jocelyndumlao/consumer-review-of-clothing-product
🧩 Embeddings Used
🔹 Word2Vec

Trained on the dataset corpus (300-dimensional vectors)

🔹 GloVe

Pretrained vectors (100D / 300D options)

🔹 FastText

Subword-aware embeddings

Trained on corpus or loaded externally

🔹 BERT (Sentence-Transformers)

Model: all-MiniLM-L6-v2

Produces 384-dimensional contextual embeddings

Expected to achieve the best performance

## 📥 Preprocessing Pipeline
Lowercase text

Tokenization (NLTK)

Remove noise and punctuation

Handle missing text fields

Generate numeric embeddings

Labels are multi-label encoded using:

Recommended_IND

Department Name

Class Name

Fallback fields (Cloth_class, Cons_rating) are used if label columns are missing.

## 🚀 Workflow Summary
1️. Data Loading & Cleaning
Lowercasing

Token filtering

Punctuation removal

Missing value handling

2️. Multi-Label Target Construction
Convert Recommended IND to binary

One-hot encode department and class fields

3️. Embedding Generation
Word2Vec (trained)

GloVe (pretrained)

FastText (trained, OOV handling)

BERT-SBERT contextual embeddings

4️. Model Training
A PyTorch neural classifier with:
fully connected layers

Loss: BCEWithLogitsLoss

Optimizer: Adam

5️. Evaluation Metrics
Micro F1

Macro F1

Precision, Recall

ROC-AUC per label

6️. Result Saving
Outputs include:

Metrics stored in a CSV file

Performance visualization plots





