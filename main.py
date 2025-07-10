import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the model
model = load_model("model/speech_disorder_model.keras")

# Load the label encoder (make sure the order matches your training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Disorder Voices', 'Normal Voices'])  # Correct class order from your training

def predict_file(file_path, model, label_encoder, n_mfcc=40, max_pad_len=174):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        pred = model.predict(mfcc)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        if pred_label == 'Normal Voices':
            print(f"Prediction: Normal Speech (No disorder detected)")
        else:
            print(f"Prediction: Speech Disorder Detected")
        print(f"Raw model output: {pred_label}")
    except Exception as e:
        print(f"[Error during processing or prediction]: {e}")

# Test on a sample file
predict_file("test_audio/653.wav", model, label_encoder)
