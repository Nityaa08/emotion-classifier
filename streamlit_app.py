
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf

MODEL_PATH = "best_mel_cnn.h5"  # ensure trained model saved here

st.title("Emotion Classifier (RAVDESS-trained)")
st.write("Upload a WAV file (speech). Will predict emotion among: neutral, happy, sad, angry.")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])
if uploaded_file is not None:
    # Save to temp and process
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    y, sr = librosa.load("temp.wav", sr=16000, mono=True)
    # preprocess similarly as in notebook
    y, _ = librosa.effects.trim(y)
    y = y / max(abs(y)) if max(abs(y))>0 else y
    # extract mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    # pad or crop to 128 frames
    import numpy as np
    def pad_spec(spec, max_len=128):
        n_mels, t = spec.shape
        if t < max_len:
            pad_width = max_len - t
            left = pad_width // 2
            right = pad_width - left
            spec_padded = np.pad(spec, ((0,0),(left,right)), mode='constant', constant_values=(spec.min(),))
        elif t > max_len:
            start = (t - max_len) // 2
            spec_padded = spec[:, start:start+max_len]
        else:
            spec_padded = spec
        return spec_padded
    S_p = pad_spec(S_db, 128)
    X = S_p[np.newaxis, ..., np.newaxis]  # shape (1, n_mels, time, 1)
    model = load_model(MODEL_PATH)
    pred = np.argmax(model.predict(X), axis=1)[0]
    label_map = {0:"neutral", 1:"happy", 2:"sad", 3:"angry"}  # set accordingly to your label encoding
    st.write("Predicted emotion:", label_map.get(pred, "unknown"))
