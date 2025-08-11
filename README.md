# 🌀 NLP Twitter Disaster Classification Project

This project classifies tweets as **disaster-related** or **not disaster-related** using **Natural Language Processing** techniques using multiple machine learning models and deep learning techniques. The final deliverable includes a Streamlit web app for interactive predictions.

## Project Overview

The goal is to build a model that can identify whether a tweet describes a disaster event. The dataset consists of tweets labeled as disaster-related or not. The project covers the full ML lifecycle:

- Data loading and inspection  
- Exploratory Data Analysis (EDA)  
- Text preprocessing and feature engineering  
- Model training with various algorithms (Logistic Regression, Naive Bayes, Random Forest, SVM, Neural Networks)  
- Hyperparameter tuning with GridSearchCV  
- Evaluation with metrics such as Accuracy, F1-score, ROC AUC, and confusion matrices  
- Final CNN model training and comparison  
- Deployment via a Streamlit app

## 📂 Project Structure
``` 
NLP_twitter_disaster_classification_project/
├── data/
│ └── train.csv # Raw dataset (from Kaggle)
├── disaster_tweet_app/
│ ├── app.py # Streamlit user interface
│ ├── models.py # Model training & prediction functions
│ └── preprocessing.py # Text cleaning & tokenization helpers
├── models/ # Saved model weights (gitignored)
├── requirements.txt # Python dependencies
└── Disaster_Tweet_Classification_NLP_Project.ipynb # Jupyter notebook for exploration and prototyping
└── README.md
--
``` 

---

## Key Features

### Data Loading & Inspection
- Load raw dataset
- Inspect size, missing values, and sample tweets

### Exploratory Data Analysis (EDA)
- Analyze tweet lengths distribution by class
- Visualize class balance

### Text Preprocessing
- Cleaning: Lowercasing, removing numbers, URLs, HTML tags, punctuation, special characters  
- Tokenization  
- POS tagging and lemmatization  
- Removal of stopwords and short words  
- Creation of custom stopword list by identifying overlapping frequent words in both classes

### Feature Engineering
- Analyze frequent words in disaster vs non-disaster tweets  
- Vectorize text using CountVectorizer and TF-IDF (max 5000 features)

### Model Training & Evaluation
- Train-test split (80-20 stratified)
- Models trained:  
  - Logistic Regression  
  - Naive Bayes  
  - Random Forest  
  - Support Vector Machines (SVM)  
  - Feed-forward Neural Network (Dense layers)
- Evaluation metrics: Accuracy, F1 score, ROC AUC, confusion matrix
- Hyperparameter tuning using GridSearchCV with n-grams and model-specific parameters

### Deep Learning Model
- Convolutional Neural Network (CNN) trained on tokenized and padded sequences of tweets  
- Compare CNN performance with classical ML models

### Streamlit App
- Interactive app to preprocess, train models, evaluate, and predict if new tweets are disaster-related or not
- Supports Logistic Regression and CNN models

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/NLP_twitter_disaster_classification_project.git
   cd NLP_twitter_disaster_classification_project
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt

3. Download the dataset (train.csv) and place it inside the data/ folder.

## Usage
Running the Jupyter Notebook
The notebook Disaster_Tweet_Classification_NLP_Project.ipynb contains a step-by-step walkthrough of EDA, preprocessing, modeling, tuning, and evaluation.

## Streamlit App
Launch the Streamlit app:
  ```bash
    streamlit run disaster_tweet_app/app.py
```
The app allows you to:

* Preprocess tweets
* Train logistic regression and CNN models
* Evaluate models
* Input new tweets to predict disaster relevance

## Results
* Models trained on both TF-IDF and CountVectorizer features with detailed evaluation reports.
* CNN achieves competitive performance by learning from word embeddings and convolutional features.
* Hyperparameter tuning improved model performance significantly.
* Final best model selected based on cross-validation F1 scores.

## Acknowledgments
* Dataset sourced from Kaggle: Real or Not? NLP with Disaster Tweets
*Tutorials and documentation from sklearn, nltk, tensorflow, and streamlit communities.

## License
This project is licensed under the MIT License.
