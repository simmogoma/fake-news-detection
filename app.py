import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

# NLTK Downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# Custom CSS for HUGE Title and Styling
st.markdown("""
    <style>
    .big-title {
        font-size: 60px !important;
        font-weight: 800 !important;
        color: #FF4B4B; /* Streamlit Red color */
        text-align: center;
        margin-top: -50px;
    }
    .sub-title {
        font-size: 24px !important;
        text-align: center;
        color: #7d7d7d;
        margin-bottom: 40px;
    }
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
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']).astype({'label': int})
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['content'])
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X, df['label'])
    return model, vectorizer, df

# --- UI Display ---
st.markdown('<p class="big-title">Normal vs K - Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced AI Analysis System</p>', unsafe_allow_html=True)

try:
    model, vectorizer, original_df = load_and_train()
    
    tab1, tab2, tab3 = st.tabs(["📝 Custom Text Analysis", "🔍 Keyword Search", "ℹ️ How it Works"])

    with tab1:
        st.subheader("Manual News Verification")
        user_input = st.text_area("Paste the news article here:", height=250)
        if st.button("Analyze News", use_container_width=True):
            if user_input:
                with st.spinner('Analyzing patterns...'):
                    cleaned = clean_text(user_input)
                    vec = vectorizer.transform([cleaned])
                    pred = model.predict(vec)
                    if pred[0] == 1:
                        st.success("### ✅ RESULT: THIS NEWS APPEARS REAL")
                    else:
                        st.error("### 🚨 RESULT: THIS NEWS APPEARS FAKE")
            else:
                st.warning("Please paste some text first!")

    with tab2:
        st.subheader("Search in Dataset")
        keyword = st.text_input("Enter a keyword (e.g., 'Election', 'Health', 'Policy'):")
        if keyword:
            results = original_df[original_df['content'].str.contains(keyword.lower())]
            if not results.empty:
                st.write(f"Found {len(results)} matches in our training data:")
                st.dataframe(results[['title', 'label']].head(10))
            else:
                st.info("No exact match found in our database.")

    with tab3:
        st.markdown("""
        ### Understanding the System
        - **Algorithm:** Passive Aggressive Classifier.
        - **Processing:** TF-IDF Vectorization for text-to-math conversion.
        - **Database:** Trained on over 5,000 verified news records.
        """)

except Exception as e:
    st.error(f"System Error: {e}")
