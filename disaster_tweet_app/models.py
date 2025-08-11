import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

# ===== Logistic Regression =====
def train_logistic_regression(df):
    X = df['text_processed']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_tfidf, y_train)
    y_pred = lr.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame([{
        "Model": "Logistic Regression",
        "Accuracy": acc,
        "F1 Score": f1
    }])

    return results, lr, tfidf


def predict_logistic(tweet, tfidf, model):
    vect = tfidf.transform([tweet])
    pred = model.predict(vect)[0]
    return pred


# ===== CNN =====

def build_cnn(max_length, vocab_size):
    model = Sequential(name='sequential')
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn(df):
    X = df['text_clean']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    max_len = max(X_train.apply(len).max(), 50)  # Minimum length 50 for padding

    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index

    model = build_cnn(max_len, vocab_size)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    model.fit(
        X_train_pad, y_train.astype('int32'),
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    loss, acc = model.evaluate(X_test_pad, y_test.astype('int32'), verbose=0)
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    f1 = f1_score(y_test, y_pred)

    results = pd.DataFrame([{
        "Model": "CNN",
        "Accuracy": acc,
        "F1 Score": f1
    }])

    # Return model, tokenizer, max_len for prediction on new data
    return results, model, tokenizer, max_len

def predict_cnn(tweet, tokenizer, model, max_len):
    seq = tokenizer.texts_to_sequences([tweet])
    pad = sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    pred_prob = model.predict(pad)[0][0]
    pred = int(pred_prob > 0.5)
    return pred
