import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Fake News AI", layout="wide")

# NLTK Setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# Custom CSS for UI
st.markdown("""
    <style>
    .main-title { font-size: 65px !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: -20px; }
    .sub-title { font-size: 22px; text-align: center; color: #808495; margin-bottom: 40px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

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
    df = df.dropna(subset=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    
    # ngram_range=(1,2) model ko behtar patterns seekhne mein help karega
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)
    
    # Calculate Accuracy
    X_test_tfidf = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    
    return model, vectorizer, acc, df

# UI Header
st.markdown('<p class="main-title">Normal vs K - News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Passive Aggressive Learning Algorithm</p>', unsafe_allow_html=True)

try:
    model, vectorizer, acc, original_df = load_and_train()
    st.sidebar.metric("System Accuracy", f"{acc*100:.2f}%")

    tab1, tab2, tab3 = st.tabs(["🔍 Analysis Center", "📊 Database Search", "📖 Instructions"])

    with tab1:
        user_input = st.text_area("Paste English News Content Here:", height=250, placeholder="Type or paste article...")
        if st.button("RUN AI VERIFICATION", use_container_width=True):
            if user_input:
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)
                
                # Confidence Score (Decision Function)
                confidence = model.decision_function(vec)[0]
                
                if prediction[0] == 1:
                    st.success(f"### ✅ RESULT: REAL NEWS")
                    st.info(f"AI Confidence Level: Positive ({confidence:.2f})")
                else:
                    st.error(f"### 🚨 RESULT: FAKE NEWS")
                    st.info(f"AI Confidence Level: Negative ({confidence:.2f})")
            else:
                st.warning("Please enter text to analyze.")

    with tab2:
        search = st.text_input("Search keywords in dataset:")
        if search:
            matches = original_df[original_df['content'].str.contains(search.lower())].head(10)
            st.table(matches[['title', 'label']])

    with tab3:
        st.write("This system uses TF-IDF Vectorization and Passive Aggressive Classification to detect linguistic patterns common in fake news.")

except Exception as e:
    st.error(f"Initialization Error: {e}")
