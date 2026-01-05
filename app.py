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
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Fake News AI Detector", layout="wide")

# --------------------------------------------------
# 2. GOOGLE GEMINI API SETUP (STREAMLIT CLOUD)
# --------------------------------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# --------------------------------------------------
# 3. NLTK SETUP (FIX-2 APPLIED)
# --------------------------------------------------
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# --------------------------------------------------
# 4. CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main-title { 
    font-size: 60px; 
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

# --------------------------------------------------
# 5. TEXT CLEANING (FIX-1 APPLIED)
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    try:
        tokens = nltk.word_tokenize(text)
    except:
        # Streamlit Cloud fallback
        tokens = text.split()

    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# 6. LOAD DATA & TRAIN MODEL
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

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(X_test_vec))

    return model, vectorizer, acc

# --------------------------------------------------
# 7. UI
# --------------------------------------------------
st.markdown('<p class="main-title">Fake News AI Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning + Google Gemini AI</p>', unsafe_allow_html=True)

model, vectorizer, acc = load_and_train()

st.sidebar.title("📊 Model Info")
st.sidebar.metric("ML Accuracy", f"{acc*100:.2f}%")
st.sidebar.write("Model: Passive Aggressive")
st.sidebar.write("AI: Google Gemini")

tab1, tab2 = st.tabs(["🔍 Analysis Center", "📖 Instructions"])

# --------------------------------------------------
# 8. ANALYSIS TAB
# --------------------------------------------------
with tab1:
    user_input = st.text_area("Paste English News Content Here:", height=250)

    if st.button("RUN AI VERIFICATION", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter news content.")
        else:
            with st.spinner("Analyzing with ML + Gemini AI..."):

                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])

                ml_pred = model.predict(vec)[0]
                ml_conf = model.decision_function(vec)[0]

                prompt = f"""
                Analyze the following news and tell whether it is REAL or FAKE.
                Give a short reason.

                News:
                {user_input}
                """
                gemini_response = gemini_model.generate_content(prompt)

                st.divider()

                if ml_pred == 1:
                    st.success("✅ ML RESULT: REAL NEWS")
                else:
                    st.error("🚨 ML RESULT: FAKE NEWS")

                st.write(f"**ML Confidence Score:** {ml_conf:.2f}")

                st.subheader("🤖 Gemini AI Opinion")
                st.info(gemini_response.text)

# --------------------------------------------------
# 9. INSTRUCTIONS TAB
# --------------------------------------------------
with tab2:
    st.write("""
    **How to use:**
    1. Paste English news text
    2. Click RUN AI VERIFICATION
    3. View ML + Gemini AI result
    """)
