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

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fake News AI Detector", layout="wide")

# --- 2. Google Gemini API Setup ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# --- 3. NLTK Setup ---
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    return True

download_nltk_resources()
stop_words = set(stopwords.words('english'))

# --- 4. Custom CSS ---
st.markdown("""
<style>
.main-title { 
    font-size: 65px; 
    font-weight: 800; 
    color: #FF4B4B; 
    text-align: center;
}
.sub-title { 
    font-size: 22px; 
    text-align: center; 
    color: #808495; 
}
</style>
""", unsafe_allow_html=True)

# --- 5. Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

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

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))

    return model, vectorizer, acc

# --- 6. UI ---
st.markdown('<p class="main-title">Fake News Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning + Google Gemini AI</p>', unsafe_allow_html=True)

model, vectorizer, acc = load_and_train()

st.sidebar.title("📊 Model Info")
st.sidebar.metric("ML Accuracy", f"{acc*100:.2f}%")
st.sidebar.write("Model: Passive Aggressive")
st.sidebar.write("AI: Google Gemini")

tab1, tab2 = st.tabs(["🔍 News Analysis", "📖 Instructions"])

# --- 7. Analysis Tab ---
with tab1:
    user_input = st.text_area("Paste English News Text:", height=250)

    if st.button("RUN AI VERIFICATION", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter news content.")
        else:
            with st.spinner("Analyzing with ML + Gemini AI..."):

                # ML Prediction
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                ml_prediction = model.predict(vec)[0]
                ml_confidence = model.decision_function(vec)[0]

                # Gemini AI Analysis
                prompt = f"""
                Analyze the following news and tell whether it is FAKE or REAL.
                Give a short reason.

                News:
                {user_input}
                """
                gemini_response = gemini_model.generate_content(prompt)

                st.divider()

                if ml_prediction == 1:
                    st.success("✅ ML RESULT: REAL NEWS")
                else:
                    st.error("🚨 ML RESULT: FAKE NEWS")

                st.write(f"**ML Confidence Score:** {ml_confidence:.2f}")

                st.subheader("🤖 Gemini AI Opinion")
                st.info(gemini_response.text)

# --- 8. Instructions Tab ---
with tab2:
    st.write("""
    **How to use this system:**
    1. Paste English news text
    2. Click RUN AI VERIFICATION
    3. View ML + Google Gemini results
    """)
