import streamlit as st
import pandas as pd
import re
import nltk
import google.generativeai as genai

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Fake News AI Detector", layout="wide")

# --------------------------------------------------
# GEMINI API SETUP
# --------------------------------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Force use of newer model names that work with the latest SDK
try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except:
    gemini_model = genai.GenerativeModel("gemini-pro")

# --------------------------------------------------
# NLTK SETUP
# --------------------------------------------------
@st.cache_resource
def setup_nltk():
    try:
        nltk.download("punkt")
        nltk.download("stopwords")
        return set(stopwords.words("english"))
    except:
        return set()

stop_words = setup_nltk()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# LOAD DATA & TRAIN MODEL
# --------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("news_data_final.csv")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(vectorizer.transform(X_test)))
    return model, vectorizer, acc

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📰 Fake News AI Detector")
st.caption("Machine Learning + Google Gemini AI")

model, vectorizer, acc = load_and_train()
st.sidebar.metric("ML Accuracy", f"{acc*100:.2f}%")

user_input = st.text_area("Paste English News Content", height=250)

if st.button("RUN AI VERIFICATION", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter news text")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            ml_pred = model.predict(vec)[0]
            ml_conf = model.decision_function(vec)[0]

            if ml_pred == 1:
                st.success("✅ ML RESULT: REAL NEWS")
            else:
                st.error("🚨 ML RESULT: FAKE NEWS")

            st.write(f"**ML Confidence:** {ml_conf:.2f}")

            st.subheader("🤖 Gemini AI Opinion")

            try:
                # Direct call - latest SDK works best with this
                response = gemini_model.generate_content(
                    f"Analyze this news and explain if it is real or fake in 2 sentences: {user_input}"
                )
                st.info(response.text)
            except Exception as e:
                # Final fallback for outdated environments
                st.error("Gemini is currently unavailable in this environment.")
                st.warning("Please reboot the app on Streamlit Cloud to update libraries.")
