# Prediction of Fake Tweets Using Machine Learning Algorithms

# Twitter Fake Tweets Detection

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Detect fake tweets using machine learning & NLP techniques — powered by Python, TensorFlow, and Scikit-learn.
> In this project, we worked upon the models to determine the
fake news among the datasets using data mining
techniques, and this paper had about five different
classification models which are compared by the
accuracies. SVM got the highest accuracy and it is about
99.48%.
Fake news recognition is an research
which has a large number of datasets. There are no
information on real time data or with respect to the current
issues. The current political dataset, indicating that the
model performs well against it.
In this project fake tweets detection we ought to detect the
fake tweets among the Political affairs, so we wanted to
work on the rumored tweets in the future, and we also
wanted to explore the data to compare the further results
based on algorithm accuracies.

---

This Team project of four was undertaken as part of my Bachelor’s Degree in Computer Science and Engineering (2020), aimed at addressing one of the most prevalent issues in the modern digital age — the spread of fake information on social media platforms.
With the explosive growth of platforms like Twitter, the ease of sharing information has increased drastically. While this democratizes content creation, it also opens the door for misinformation and disinformation to spread at unprecedented speeds. Fake news can manipulate public opinion, influence political discourse, and even cause real-world harm.

##  Overview

The project addresses the **fake news epidemic** on social media, focusing on **Twitter**.  
Twitter is both a hub for personal opinions and official announcements, making it a prime target for misinformation.  
Our mission: **Predict whether a tweet is real or fake** using advanced data mining and classification algorithms.
The main objective of this project is to design and implement a machine learning-based system that can classify tweets as real or fake based on their content and associated metadata. Using a Kaggle dataset of tweets, the project focuses on:

Data Preprocessing: Cleaning, normalizing, and structuring the data to improve machine learning performance.
Natural Language Processing (NLP): Using tokenization, stemming, stop-word removal, and vectorization to transform text into meaningful features.
Model Training & Evaluation: Applying and comparing multiple classification algorithms such as Logistic Regression, Naïve Bayes, Stochastic Gradient Descent, Support Vector Machine, and Random Forest.
Performance Analysis: Measuring accuracy and comparing results across different algorithms to identify the most effective model.
This project not only demonstrates the application of data mining, NLP, and machine learning in solving real-world problems but also lays a foundation for future research in automated misinformation detection.


---

## Dataset
We used a Kaggle dataset containing: Sentement.csv, true.csv, fake.csv
- **Metadata:** `id`, `candidate`, `sentiment`, `subject_matter`
- **Engagement Data:** `retweet_count`, `tweet_location`, `user_timezone`
- **Text Content:** `text of tweets for NLP processing`

---

## Data Preprocessing
We applied **deep learning-powered preprocessing** to clean and structure the data:
-  **Cleaning:** Removed missing values & noise
-  **Tokenization & Stemming:** Used NLTK to break tweets into tokens & reduce words to root form
-  **Lowercasing & Stop Words Removal:** Standardized and removed common filler words
-  **Vectorization:** Applied Count Vectorizer to convert text into numerical features

---

##  Methodology
We trained multiple classifiers and compared results:

| Algorithm | Pros  | Cons  |
|-----------|--------|---------|
| **Logistic Regression** | Simple, fast, easy to interpret | Linear assumption |
| **Naïve Bayes** | Great for text data, fast | Lower accuracy in this case |
| **SGD (Stochastic Gradient Descent)** | Fast, memory efficient | Noisy convergence |
| **SVM (Support Vector Machine)** | High accuracy, works well in high-dim spaces | Slow on large datasets |
| **Random Forest** | Robust, handles missing data | Computationally heavy |

---

## Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | **98.68%** |
| Naïve Bayes | **94.08%** |
| Stochastic Gradient Descent (SGD) | **99.27%** |
| Support Vector Machine (SVM) | **99.48%** |
| Random Forest | **98.45%** |

 **Top Performers:** SVM & SGD  
 **Lowest Accuracy:** Naïve Bayes  
 **Impact:** Data preprocessing significantly boosted accuracy.

---

## Installation & Usage
```bash
# Clone the repo
git clone https://github.com/yourusername/twitter-fake-tweets-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py




