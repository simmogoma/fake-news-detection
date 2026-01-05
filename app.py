import streamlit as st
import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")
    df['content'] = df['title'] + " " + df['text']   # 🔥 KEY FIX
    return df

df = load_data()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if w not in stopwords.words('english') and len(w) > 2]
    return " ".join(text)

df['content'] = df['content'].apply(clean_text)

X = df['content']
y = df['label']

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, acc

model, vectorizer, accuracy = train_model()

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

user_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter news text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction.upper() == "FAKE":
            st.error("❌ This news is FAKE")
        else:
            st.success("✅ This news is REAL")
