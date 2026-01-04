import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fake News AI Detector", layout="wide")

# --- 2. NLTK Setup (Error Fix) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk_data()
stop_words = set(stopwords.words('english'))

# --- 3. Custom CSS (Bada Title aur Styling) ---
st.markdown("""
    <style>
    .main-title { 
        font-size: 70px !important; 
        font-weight: 800 !important; 
        color: #FF4B4B; 
        text-align: center; 
        margin-top: -40px;
        margin-bottom: 0px; 
    }
    .sub-title { 
        font-size: 24px !important; 
        text-align: center; 
        color: #808495; 
        margin-bottom: 40px; 
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 24px; 
        justify-content: center; 
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_and_train():
    # File load karein (Ensure 10k balanced file is on GitHub)
    df = pd.read_csv('news_data_final.csv')
    
    # Data preprocessing
    df['content'] = (df['title'].fillna('') + " " + df['text'].fillna('')).apply(clean_text)
    df = df.dropna(subset=['label'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorization (ngram_range adds better context)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Model Training
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)
    
    # Accuracy calculation
    X_test_tfidf = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    
    return model, vectorizer, acc, df

# --- 5. App UI ---
st.markdown('<p class="main-title">Normal vs K - News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Machine Learning Analysis</p>', unsafe_allow_html=True)

try:
    # Model loading
    model, vectorizer, acc, original_df = load_and_train()
    
    # Sidebar stats
    st.sidebar.title("📊 Model Analytics")
    st.sidebar.metric("System Accuracy", f"{acc*100:.2f}%")
    st.sidebar.info("Model: Passive Aggressive Classifier")
    
    # Tabs layout
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis Center", "📊 Database Search", "📖 Instructions"])

    with tab1:
        st.subheader("Verify News Authenticity")
        user_input = st.text_area("Paste the news article text here:", height=250, placeholder="Example: Scientists discover life on Mars...")
        
        if st.button("RUN AI VERIFICATION", use_container_width=True):
            if user_input:
                with st.spinner('AI is analyzing linguistic patterns...'):
                    cleaned = clean_text(user_input)
                    vec = vectorizer.transform([cleaned])
                    prediction = model.predict(vec)
                    
                    # Confidence Score
                    confidence = model.decision_function(vec)[0]
                    
                    st.divider()
                    if prediction[0] == 1:
                        st.success("### ✅ RESULT: THIS NEWS IS REAL")
                        st.write(f"**AI Confidence Score:** {confidence:.2f} (Positive score indicates Real)")
                    else:
                        st.error("### 🚨 RESULT: THIS NEWS IS FAKE")
                        st.write(f"**AI Confidence Score:** {confidence:.2f} (Negative score indicates Fake)")
            else:
                st.warning("⚠️ Please paste some news content first.")

    with tab2:
        st.subheader("Search Training Dataset")
        search_query = st.text_input("Enter a keyword to find related records:")
        if search_query:
            results = original_df[original_df['content'].str.contains(search_query.lower())].head(15)
            if not results.empty:
                st.dataframe(results[['title', 'label']], use_container_width=True)
            else:
                st.info("No records found for this keyword.")

    with tab3:
        st.markdown("""
        ### How it Works?
        1. **TF-IDF Vectorization:** Convert text into numerical patterns.
        2. **N-Grams (1,2):** Analyzes single words and pairs of words.
        3. **Passive Aggressive Classifier:** A high-speed algorithm ideal for large scale text classification.
        4. **Dataset:** Trained on 10,000 balanced records of verified news.
        """)

except Exception as e:
    st.error(f"❌ System Initialization Error: {e}")
