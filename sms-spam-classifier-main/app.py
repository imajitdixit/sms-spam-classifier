import streamlit as st

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

download_nltk_data()

ps = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in STOP_WORDS and w not in string.punctuation]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_artifacts()

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning(" Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")

