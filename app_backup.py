import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("saved_model")
    model = DistilBertForSequenceClassification.from_pretrained("saved_model")
    return tokenizer, model

tokenizer, model = load_model()

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1E88E5;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered News Verification using DistilBERT</p>',
    unsafe_allow_html=True
)

# Input Area
news_text = st.text_area(
    "Paste News Article or Headline",
    height=200,
    placeholder="Enter news text here..."
)

# Predict Button
if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        with st.spinner("Analyzing news..."):

            inputs = tokenizer(
                news_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )

            with torch.no_grad():
                outputs = model(**inputs)

            prediction = torch.argmax(outputs.logits, dim=1).item()

            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence = torch.max(probabilities).item() * 100

        st.divider()

        if prediction == 1:
            st.success("✅ REAL NEWS")
            st.metric("Confidence Score", f"{confidence:.2f}%")
        else:
            st.error("❌ FAKE NEWS")
            st.metric("Confidence Score", f"{confidence:.2f}%")

# Footer
st.divider()
st.caption("Developed using DistilBERT • Fake News Detection Project")