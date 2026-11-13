"""
Streamlit version of the demo. Run: streamlit run streamlit_app.py
(The Gradio version in app.py is the one used for the Hugging Face Space.)
"""
import os
import tempfile

import streamlit as st

from emotion import predict

st.title("Emotion Classifier (RAVDESS-trained)")
st.write("Upload a WAV file of speech. Predicts one of: angry, happy, neutral, sad.")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])
if uploaded_file is not None:
    # A per-upload temporary file, so two users never overwrite each other's audio.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(uploaded_file.getbuffer())
        path = f.name
    try:
        probs = predict(path)
    except ValueError as e:
        st.error(str(e))
    else:
        best = max(probs, key=probs.get)
        st.write("Predicted emotion:", f"**{best}** ({probs[best]:.0%})")
        st.bar_chart(probs)
    finally:
        os.remove(path)
