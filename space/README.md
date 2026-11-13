---
title: Speech Emotion Classifier
emoji: 🎙️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: CNN on log-mel spectrograms, trained on RAVDESS speech
---

# Speech Emotion Classifier

Upload or record a few seconds of speech; a small CNN on log-mel spectrograms predicts angry, happy, neutral or sad.
Trained on 4 emotions of the RAVDESS speech set; 68% accuracy on a held-out 20% split that shares actors with training.

Code, training notebook and limits: https://github.com/Nityaa08/emotion-classifier · by [Nityashree S](https://nityashree.in)
