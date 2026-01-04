import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# NLTK Downloads shuru mein hi karein
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
    # File ka naam wahi rakhein jo GitHub par hai
    df = pd.read_csv('news_data_final.csv') 
    
    # Data Cleaning
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = (df['title'] + " " + df['text']).apply(clean_text)
    
    # Label formatting
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']).astype({'label': int})
    
    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=7)
    
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(X_train)
    
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)
    
    return model, vectorizer

# --- App Interface ---
st.title("Fake News Detection System")

# Model load karein
try:
    model, vectorizer = load_and_train()
    
    user_input = st.text_area("Enter News Article Content here:")
    if st.button("Predict"):
        if user_input:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)
            
            # Result dikhayein
            result = "REAL" if prediction[0] == 1 else "FAKE"
            st.subheader(f"The news is: {result}")
        else:
            st.warning("Please enter some text.")
except Exception as e:
    st.error(f"Error loading data: {e}")

