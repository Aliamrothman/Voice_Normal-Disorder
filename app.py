import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load model and label encoder
@st.cache_resource

def load_model_and_encoder():
    model = load_model("model/speech_disorder_model.keras")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['Disorder Voices', 'Normal Voices'])
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

st.title("Speech Disorder Detection")
st.write("Upload a .wav audio file to test if the speech is normal or has a disorder.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    try:
        audio, sr = librosa.load("temp.wav", sr=None)
        n_mfcc = 40
        max_pad_len = 174
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        pred = model.predict(mfcc)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        # Show probabilities for each class
        st.subheader("Class Probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            st.write(f"{class_name}: {pred[0][i]:.4f}")
        # Show bar chart
        st.bar_chart({name: prob for name, prob in zip(label_encoder.classes_, pred[0])})
        if pred_label == 'Normal Voices':
            st.success("Prediction: Normal Speech (No disorder detected)")
        else:
            st.error("Prediction: Speech Disorder Detected")
        st.write(f"Raw model output: {pred_label}")
    except Exception as e:
        st.error(f"Error during processing or prediction: {e}")
    os.remove("temp.wav") 