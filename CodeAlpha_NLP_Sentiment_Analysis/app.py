import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🔥 Sentiment Analysis App")

user_input = st.text_area("Enter your text")

if st.button("Predict"):
    cleaned = user_input.lower()
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    
    st.write("Sentiment:", prediction[0])