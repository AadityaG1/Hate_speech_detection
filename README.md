# Hate Speech Detection using Logistic Regression and Sentence Embeddings

## 🔍 Project Overview

This project focuses on detecting hate speech and offensive language in short text (tweets) using a series of increasingly refined machine learning models. The primary goal is to build a **fast, accurate**, and **interpretable** classifier that can distinguish between:

- **Hate speech**
- **Offensive language**
- **Neutral (clean) content**

Here, the well-known **T. Davidson hate speech dataset** is used and three progressively improved models are implemented to balance performance and training efficiency — ideal for environments where training must complete quickly.

---


## 🧠 Approaches Used

### 1️⃣ Baseline: Logistic Regression with TF-IDF
- Vectorized tweets using `TfidfVectorizer`
- Trained a simple Logistic Regression model
- Fast training, good generalization, but limited in detecting nuanced hate speech

### 2️⃣ Improved: Sentence Embeddings + Logistic Regression
- Used pretrained sentence embeddings (`all-MiniLM-L6-v2` from `sentence-transformers`)
- Captures deeper semantic meaning than TF-IDF
- Logistic Regression on 384-dim semantic vectors
- Improved accuracy, especially for borderline neutral/offensive content

### 3️⃣ Final: Balanced Sentence Embeddings with Class Weights
- Applied class weighting in Logistic Regression to address class imbalance
- Boosted recall for underrepresented "hate" class
- Achieved significant performance improvement on "hate" recall 

---

## 🧪 Dataset

- **Source**: [T. Davidson et al., 2017](https://github.com/t-davidson/hate-speech-and-offensive-language)
- Each sample includes:
  - `text`: Tweet content
  - `label`: Class (`0 = hate`, `1 = offensive`, `2 = neutral`)

---


## 💡 Key Learnings:
- Semantic embeddings vastly improve classification compared to bag-of-words models.

- Class imbalance significantly affects hate speech detection and must be addressed.

- Logistic Regression, when paired with strong features and weights, is both interpretable and high-performing.

---
