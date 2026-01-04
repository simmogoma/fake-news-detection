# save as normal_vs_k_ui.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests
from newspaper import Article

# ------------------------
# NLTK Setup
# ------------------------
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ------------------------
# Load Dataset & Train Model
# ------------------------
@st.cache_data
def load_and_train():
# 1. File load karein (check karein GitHub par naam yahi hai na?)
    df = pd.read_csv('news_data_final.csv') 

# 2. Column names ko handle karein (title aur text ko jod kar 'content' banayein)
# Dhyan dein: Aapke CSV mein 'title' aur 'text' columns hone chahiye
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + " " + df['text']

# 3. Text cleaning apply karein
    df['content'] = df['content'].apply(clean_text)

# 4. Label ko sahi format mein layein
if 'label' in df.columns:
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    X = df['content']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

model, vectorizer = load_and_train()

# ------------------------
# Fake News Prediction
# ------------------------
def check_news(news_input):
    processed = clean_text(news_input)
    vec = vectorizer.transform([processed])
    res = model.predict(vec)
    return "REAL" if res[0] == 0 else "FAKE"

# ------------------------
# Fetch News by Keyword (NewsAPI)
# ------------------------
API_KEY = "0a25200ddae640ab9b0df13c819df07c"

def fetch_news_by_keyword(keyword, page_size=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={keyword}&language=en&sortBy=relevancy&pageSize={page_size}&apiKey={API_KEY}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            return []
        articles = [
            {"title": a["title"], "description": a["description"] or ""}
            for a in data["articles"]
        ]
        return articles
    except:
        return []

# ------------------------
# Fetch News from URL
# ------------------------
def fetch_news_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + " " + article.text
    except:
        return ""

# ------------------------
# Streamlit App Layout
# ------------------------
st.set_page_config(page_title="Normal vs K - Fake News Detector", page_icon="📰", layout="wide")
st.markdown("""
    <div style='text-align: center;'>
        <h1>📰 Normal vs K - Fake News Detector</h1>
        <p>Check if news is REAL or FAKE using AI</p>
    </div>
""", unsafe_allow_html=True)

# ------------------------
# Tabs for Navigation
# ------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Custom Text", "Keyword Search", "URL Fetch", "Instructions"])

# ---- Tab 1: Custom Text ----
with tab1:
    st.subheader("Test Custom News Text")
    news_text = st.text_area("Paste news text here:")
    if st.button("Check News", key="btn1"):
        if news_text.strip():
            result = check_news(news_text)
            if result == "REAL":
                st.success(f"This news is {result}")
            else:
                st.error(f"This news is {result}")
        else:
            st.warning("Please enter news text!")

# ---- Tab 2: Keyword Search ----
with tab2:
    st.subheader("Fetch News by Keyword")
    keyword = st.text_input("Enter keyword to search news", key="kw")
    if st.button("Fetch & Check", key="btn2"):
        if keyword.strip():
            articles = fetch_news_by_keyword(keyword)
            if articles:
                for i, art in enumerate(articles, 1):
                    content = (art['title'] or '') + " " + (art['description'] or '')
                    result = check_news(content)
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown(f"**Article {i}: {art['title']}**")
                    with col2:
                        if result == "REAL":
                            st.success(result)
                        else:
                            st.error(result)
            else:
                st.warning("No news found.")
        else:
            st.warning("Please enter a keyword!")

# ---- Tab 3: URL Fetch ----
with tab3:
    st.subheader("Fetch News from URL")
    url = st.text_input("Enter news URL", key="url")
    if st.button("Fetch & Check", key="btn3"):
        if url.strip():
            content = fetch_news_from_url(url)
            if content:
                result = check_news(content)
                if result == "REAL":
                    st.success(f"This news is {result}")
                else:
                    st.error(f"This news is {result}")
            else:
                st.warning("Could not fetch article.")
        else:
            st.warning("Please enter a URL!")

# ---- Tab 4: Instructions ----
with tab4:
    st.subheader("How to Use")
    st.markdown("""
    1. **Custom Text**: Paste any news text to check if it's real or fake.
    2. **Keyword Search**: Enter a keyword, fetch top news articles from NewsAPI, and check each article.
    3. **URL Fetch**: Paste a news URL to fetch the article text and predict.
    4. **Interpret Results**: 
        - Green = REAL news  
        - Red = FAKE news
    """)
    st.markdown("**Note:** Make sure to replace `API_KEY` with your valid NewsAPI key for keyword search.")






