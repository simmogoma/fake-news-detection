import streamlit as st
import pandas as pd
import re
import nltk
from google import genai

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News AI Detector",
    layout="wide"
)

# --------------------------------------------------
# 2. GEMINI API SETUP (NEW SDK – FIXED)
# --------------------------------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=GOOGLE_API_KEY)

# --------------------------------------------------
# 3. NLTK SETUP
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
    font-size: 58px;
    font-weight: 800;
    color: #ff4b4b;
    text-align: center;
}
.sub-title {
    font-size: 22px;
    text-align: center;
    color: #808495;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 5. TEXT CLEANING FUNCTION
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
# 6. LOAD DATA & TRAIN ML MODEL
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

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    return model, vectorizer, accuracy

# --------------------------------------------------
# 7. UI HEADER
# --------------------------------------------------
st.markdown('<p class="main-title">Fake News AI Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning + Google Gemini AI</p>', unsafe_allow_html=True)

model, vectorizer, acc = load_and_train()

# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------
st.sidebar.title("📊 Model Info")
st.sidebar.metric("ML Accuracy", f"{acc*100:.2f}%")
st.sidebar.write("Model: Passive Aggressive Classifier")
st.sidebar.write("AI: Google Gemini 1.5 Flash")

# --------------------------------------------------
# 9. TABS
# --------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Analysis Center", "📖 Instructions"])

# --------------------------------------------------
# 10. ANALYSIS TAB
# --------------------------------------------------
with tab1:
    user_input = st.text_area(
        "Paste English News Content Here:",
        height=250
    )

    if st.button("RUN AI VERIFICATION", use_container_width=True):

        if user_input.strip() == "":
            st.warning("Please enter news content.")
        else:
            with st.spinner("Analyzing with ML + Gemini AI..."):

                # ---- ML Prediction ----
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])

                ml_pred = model.predict(vec)[0]
                ml_conf = model.decision_function(vec)[0]

                # ---- Gemini Prompt ----
                prompt = f"""
Analyze the following news and determine whether it is REAL or FAKE.
Give a short and clear reason.

News:
{user_input}
"""

                # ---- Gemini API Call (NEW & FIXED) ----
                gemini_response = client.models.generate_content(
                    model="models/gemini-1.5-flash",
                    contents=prompt
                )

                st.divider()

                # ---- ML Result ----
                if ml_pred == 1:
                    st.success("✅ ML RESULT: REAL NEWS")
                else:
                    st.error("🚨 ML RESULT: FAKE NEWS")

                st.write(f"**ML Confidence Score:** {ml_conf:.2f}")

                # ---- Gemini Result ----
                st.subheader("🤖 Gemini AI Opinion")

                if gemini_response and hasattr(gemini_response, "text"):
                    st.info(gemini_response.text)
                else:
                    st.warning("No response received from Gemini AI.")

# --------------------------------------------------
# 11. INSTRUCTIONS TAB
# --------------------------------------------------
with tab2:
    st.write("""
### How to Use
1. Paste English news text in the input box
2. Click **RUN AI VERIFICATION**
3. View:
   - ML model prediction
   - Gemini AI explanation
""")
