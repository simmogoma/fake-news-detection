import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom Branding CSS
st.markdown("""
    <style>
    .main-title { font-size: 45px; font-weight: bold; color: #FFFFFF; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 18px; text-align: center; margin-bottom: 30px; color: #BBBBBB; }
    </style>
    """, unsafe_allow_html=True)

# NLTK Setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_and_train():
    df = pd.read_csv('news_data_final.csv') 
    df['content'] = (df['title'].fillna('') + " " + df['text'].fillna('')).apply(clean_text)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']).astype({'label': int})
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['content'])
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X, df['label'])
    return model, vectorizer

# --- UI Interface ---
st.markdown('<p class="main-title">📰 Normal vs K - Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Check if news is REAL or FAKE using AI</p>', unsafe_allow_html=True)

try:
    model, vectorizer = load_and_train()
    
    # Tabs create karna (Second pic jaisa)
    tab1, tab2, tab3, tab4 = st.tabs(["Custom Text", "Keyword Search", "URL Fetch", "Instructions"])

    with tab1:
        st.subheader("Test Custom News Text")
        user_input = st.text_area("Paste news text here:", height=200)
        if st.button("Check News"):
            if user_input:
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)
                result = "✅ REAL NEWS" if pred[0] == 1 else "🚨 FAKE NEWS"
                st.success(result) if pred[0] == 1 else st.error(result)
            else:
                st.warning("Please enter text.")

    with tab2:
        st.info("Keyword search feature coming soon!")

    with tab3:
        st.info("URL fetching feature coming soon!")

    with tab4:
        st.write("1. Paste any news article.\n2. AI will analyze the language patterns.\n3. Result will show if it matches real or fake news data.")

except Exception as e:
    st.error(f"Error: {e}")
