import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


st.set_page_config(page_title="NLP Sentiment App", page_icon="💬", layout="centered")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# 
st.title("💬 Advanced Sentiment Analysis App")

st.write("Enter your review below:")

text = st.text_area("Review Input")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text first.")
    else:
        processed = text.lower()
        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = np.max(prob) * 100

    
        if prediction == 1:
            st.success("😊 Positive Sentiment")
            emoji = "😊"
        else:
            st.error("😞 Negative Sentiment")
            emoji = "😞"

        
        st.subheader("Confidence Score")
        fig, ax = plt.subplots()
        ax.bar(["Confidence"], [confidence])
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        st.info(f"Model Confidence: {confidence:.2f}% {emoji}")

        
        st.subheader("Word Cloud")

        wc = WordCloud(width=500, height=300, background_color="white").generate(processed)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)