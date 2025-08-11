# Disaster Tweet Classification App

This is a Streamlit web app to classify tweets as disaster-related or not using NLP and Machine Learning models.

---

## Features

- Data preprocessing: cleaning, tokenization, and lemmatization
- Train and evaluate two models:
  - Logistic Regression
  - Convolutional Neural Network (CNN)
- Real-time tweet classification using trained models

---

## How to Run

1. **Clone the repo:**

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>/disaster_tweet_app
   
2. **Install dependencies:**
    ```bash
   pip install -r requirements.txt

4. **Download NLTK data (only once):**
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

4. **Run the app:**
   ```bash
   streamlit run app.py

5. **Use the app UI:**
   * Optionally show raw data
   * Preprocess data
   * Train Logistic Regression or CNN models
   * Input new tweets for prediction
  
## App Structure
* app.py — main Streamlit app
* preprocessing.py — text cleaning and tokenization functions
* models.py — model training and prediction functions
* data/train.csv — dataset (make sure it is in place)


