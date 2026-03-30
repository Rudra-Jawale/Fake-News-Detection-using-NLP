import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords (runs only first time)
nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Load model and vectorizer (cached so it loads only once)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Same preprocessing function from your notebook
def preprocess(text):
    stop_words = stopwords.words('english')
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# --- UI ---
st.title("🔍 Fake News Detector")
st.caption("Powered by TF-IDF + Logistic Regression · 92.2% accuracy on test data")

st.divider()

# Model stats
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "92.2%")
col2.metric("Training Samples", "6,335")
col3.metric("TF-IDF Features", "5,000")

st.divider()

# Text input
news_text = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter a news article to check if it's real or fake..."
)

# Analyze button
if st.button("Analyze Article", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("Please enter some text first.")
    elif len(news_text.split()) < 10:
        st.warning("Please enter at least 10 words for a reliable prediction.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = preprocess(news_text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]

            fake_prob = round(probabilities[0] * 100, 1)
            real_prob = round(probabilities[1] * 100, 1)

        st.divider()

        # Result
        if prediction == 0:
            st.error(f"## ⚠️ FAKE NEWS  —  {fake_prob}% confidence")
        else:
            st.success(f"## ✅ REAL NEWS  —  {real_prob}% confidence")

        # Confidence breakdown
        st.write("**Confidence Breakdown:**")
        st.progress(fake_prob / 100, text=f"Fake: {fake_prob}%")
        st.progress(real_prob / 100, text=f"Real: {real_prob}%")