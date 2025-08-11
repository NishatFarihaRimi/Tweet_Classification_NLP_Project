import streamlit as st
import pandas as pd
import os
from preprocessing import clean_text, tokenize_lemmatize
from models import train_logistic_regression, predict_logistic, train_cnn, predict_cnn

# Fix data path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'data', 'train.csv')

st.title("Disaster Tweet Classification")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# Load data once and save in session state
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = load_data(data_path)

df = st.session_state['df_original']

if st.checkbox("Show raw data"):
    st.write(df.head())

st.markdown("### Preprocessing Data")

if st.button("Run Preprocessing"):
    df_proc = df.copy()
    df_proc['text_clean'] = df_proc['text'].apply(clean_text)
    df_proc['text_processed'] = df_proc['text_clean'].apply(tokenize_lemmatize)
    st.session_state['df_processed'] = df_proc
    st.success("Preprocessing done!")
    st.write(df_proc[['text', 'text_processed']].sample(5))

st.markdown("### Model Training & Evaluation")

model_choice = st.selectbox("Select model to train:", ["Logistic Regression", "CNN"])

train_button = st.button("Train Selected Model")

if train_button:
    if 'df_processed' not in st.session_state:
        st.error("Please preprocess the data first!")
    else:
        df_train = st.session_state['df_processed']
        if model_choice == "Logistic Regression":
            results, model, tfidf = train_logistic_regression(df_train)
            st.write("Training complete!")
            st.dataframe(results)
            st.session_state['tfidf'] = tfidf
        else:
            results, model, tokenizer, max_len = train_cnn(df_train)
            st.write("Training complete!")
            st.dataframe(results)
            st.session_state['tokenizer'] = tokenizer
            st.session_state['max_len'] = max_len

        st.session_state['model'] = model
        st.session_state['model_choice'] = model_choice

st.markdown("### Predict on New Tweet")
tweet_input = st.text_area("Enter a tweet text here:")

if st.button("Predict"):
    if 'model' not in st.session_state:
        st.error("Please train a model first!")
    elif tweet_input.strip() == "":
        st.error("Please enter a tweet to predict!")
    else:
        clean_tweet = clean_text(tweet_input)
        processed_tweet = tokenize_lemmatize(clean_tweet)

        if st.session_state['model_choice'] == "Logistic Regression":
            pred = predict_logistic(processed_tweet, st.session_state['tfidf'], st.session_state['model'])
        else:
            pred = predict_cnn(clean_tweet, st.session_state['tokenizer'], st.session_state['model'], st.session_state['max_len'])

        label_map = {0: "Not Disaster", 1: "Disaster"}
        st.write(f"Prediction: **{label_map[pred]}**")

st.markdown("### End of Demo")
