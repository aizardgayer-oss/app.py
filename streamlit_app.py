import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string

# ============================================================
#                APP CONFIGURATION
# ============================================================

st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.title("🎭 Emotion Classification Using Bi-LSTM + GloVe")
st.write("Enter a text below and the model will predict the most likely emotion.")

# ============================================================
#                LOAD MODEL & TOKENIZER
# ============================================================

MODEL_PATH = "best_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 50  # must match your training setting

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# ============================================================
#                   EMOTION LABELS
# ============================================================

# Replace this list with your actual 27-class labels from your training
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# ============================================================
#                 TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)             # remove links
    text = re.sub(r"@\w+", "", text)                # remove mentions
    text = re.sub(r"#\w+", "", text)                # remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return padded

# ============================================================
#                 PREDICTION FUNCTION
# ============================================================

def predict_emotion(text):
    processed = preprocess(text)
    probs = model.predict(processed)[0]
    top_index = np.argmax(probs)
    return emotion_labels[top_index], probs

# ============================================================
#                    STREAMLIT UI
# ============================================================

user_text = st.text_area("Enter your text here:", height=120)

if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing emotion..."):
            label, probs = predict_emotion(user_text)

        st.markdown(f"### 📌 Predicted Emotion: **{label.upper()}**")

        # Display probability chart
        st.subheader("Prediction Probabilities")
        chart_data = {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}
        st.bar_chart(chart_data)

# Footer
st.markdown("---")
st.markdown("Developed using **Bi-LSTM + GloVe embeddings** | Streamlit Deployment")


