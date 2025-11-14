import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

model = load_model("lstm_model.h5")

max_len = 200   # should match your training code

st.title("NLP Text Classification App")
st.write("Enter text and the model will classify it using LSTM.")

# Input box
user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len)

        prediction = model.predict(padded)
        class_id = np.argmax(prediction)

        st.success(f"Predicted Class: {class_id}")

        st.write("### Model Confidence Scores")
        st.json({f"class_{i}": float(prediction[0][i]) for i in range(len(prediction[0]))})

# Show metrics section
st.sidebar.header("Model Performance Metrics")
st.sidebar.write("**Accuracy:** 0.58")
st.sidebar.write("**F1 Score:** 0.58")
st.sidebar.write("**Precision:** (your value)")
st.sidebar.write("**Recall:** (your value)")

