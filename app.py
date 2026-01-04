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

# --- 2. NLTK Setup ---
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
    # File load karein
    df = pd.read_csv('news_data_final.csv')
    
    # Data preprocessing
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = (df['title'] + " " + df['text']).apply(clean_text)
    
    # Label formatting
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']).astype({'label': int})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_train = vectorizer.fit_transform(X_train)
    
    # Model Training
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)
    
    # Accuracy calculation
    tfidf_test = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(tfidf_test))
    
    return model, vectorizer, acc

# --- 5. App UI ---
st.markdown('<p class="main-title">Normal vs K - News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Machine Learning Analysis</p>', unsafe_allow_html=True)

try:
    model, vectorizer, acc = load_and_train()
    
    # Sidebar Accuracy Metric
    st.sidebar.title("📊 Model Analytics")
    st.sidebar.metric("System Accuracy", f"{acc*100:.2f}%")
    st.sidebar.write("Algorithm: Passive Aggressive")
    st.sidebar.info("Model is trained on balanced data (50% Real / 50% Fake)")

    # Tabs layout (Ab sirf 2 tabs bache hain)
    tab1, tab2 = st.tabs(["🔍 Analysis Center", "📖 Instructions"])

    with tab1:
        st.subheader("Verify News Article")
        user_input = st.text_area("Paste English News Content Here:", height=250, placeholder="Paste your article text here to verify its authenticity...")
        
        if st.button("RUN AI VERIFICATION", use_container_width=True):
            if user_input:
                with st.spinner('Analyzing linguistic patterns...'):
                    cleaned = clean_text(user_input)
                    vec = vectorizer.transform([cleaned])
                    prediction = model.predict(vec)
                    
                    # Confidence Score
                    confidence = model.decision_function(vec)[0]
                    
                    st.divider()
                    if prediction[0] == 1:
                        st.success("### ✅ RESULT: THIS NEWS IS REAL")
                        st.write(f"**AI Confidence Score:** {confidence:.2f}")
                    else:
                        st.error("### 🚨 RESULT: THIS NEWS IS FAKE")
                        st.write(f"**AI Confidence Score:** {confidence:.2f}")
            else:
                st.warning("Please enter some text first.")

    with tab2:
        st.markdown("""
        ### Instructions
        1. **Copy** a news article from a website or social media.
        2. **Paste** the text into the 'Analysis Center' text box.
        3. Click the **'RUN AI VERIFICATION'** button.
        4. The system will analyze word patterns and tell you if it's likely **Real** or **Fake**.
        
        ---
        **Note:** This model is optimized for English text. Accuracy may vary for very short sentences.
        """)

except Exception as e:
    st.error(f"Error loading data: {e}")
