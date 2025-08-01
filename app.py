# app.py - Streamlit Sentiment Analysis Web App

import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#  Page Config 
st.set_page_config(page_title="Amazon Review Sentiment", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #2c2c2c;
    }

    .main-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    .stButton > button {
        background-color: #ff7f50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #e2663f;
    }
    </style>
""", unsafe_allow_html=True)


# Your content
st.title("📦 Sentiment Analysis for Product Reviews")

# Prediction logic here...

st.markdown('</div>', unsafe_allow_html=True)


# Load dataset from CSV
data = pd.read_csv("flipkart_amazon_reviews_sample.csv")

# Check if correct column names exist
if 'Review' not in data.columns or 'Sentiment' not in data.columns:
    st.error("Dataset must contain 'Review' and 'Sentiment' columns.")
    st.stop()


# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    return ' '.join(words)

data['Cleaned_Review'] = data['Review'].apply(clean_text)

# Vectorize and train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Cleaned_Review'])
y = data['Sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
model = MultinomialNB()
model.fit(X, y)

# Streamlit app layout
#st.title("Sentiment Analysis with Naive Bayes")
st.write("Enter a product review and click Analyze to see the sentiment.")

# User input
user_input = st.text_input("Write a review here:", key="review_input")



if st.button("Analyze"):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]

    # Map back to label
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    result = label_map[prediction]

    st.subheader("Predicted Sentiment:")
    st.success(result)
