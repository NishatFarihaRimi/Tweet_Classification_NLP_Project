import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def tokenize_lemmatize(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 1]
    return " ".join(words)
