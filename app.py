"""
Gradio demo (for a Hugging Face Space): upload or record speech, get the
predicted emotion among angry, happy, neutral and sad.
Run locally: pip install -r requirements.txt && python app.py
"""
import gradio as gr

from emotion import LABELS, predict

DESCRIPTION = """
A small CNN on log-mel spectrograms, trained on the RAVDESS speech set (4 of its 8 emotions,
672 clips from 24 actors). It reached **68% accuracy on a held-out 20% split** (135 clips).
That split shares actors with training, so expect lower accuracy on new voices, accents and
microphones. RAVDESS is acted North American English speech. By [Nityashree S](https://nityashree.in) ·
[Code](https://github.com/Nityaa08/emotion-classifier)
"""


def classify(audio_path):
    if not audio_path:
        raise gr.Error("Upload or record a clip first.")
    try:
        return predict(audio_path)
    except ValueError as e:
        raise gr.Error(str(e))


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech (a few seconds)"),
    outputs=gr.Label(num_top_classes=len(LABELS), label="Predicted emotion"),
    title="Speech Emotion Classifier",
    description=DESCRIPTION,
    flagging_mode="never",
)

# Neutral theme to match the author's other demos.
THEME = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "system-ui", "sans-serif"],
)

if __name__ == "__main__":
    demo.launch(theme=THEME)
