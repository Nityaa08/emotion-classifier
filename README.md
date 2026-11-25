# Speech Emotion Classifier

**Try it:** run the Gradio demo locally in a minute ([Run it](#run-it)), or deploy it free on Hugging Face Spaces with the files in [`space/`](space/) ([Deploy](#deploy-to-hugging-face-spaces)) · by [Nityashree S](https://nityashree.in)

Classifies a short clip of speech as **angry, happy, neutral or sad** with a small convolutional network over log-mel spectrograms, trained on the [RAVDESS](https://zenodo.org/record/1188976) speech set.

![Gradio demo: upload or record speech, see the probability of each emotion](docs/screenshot.png)

## What it does and why

Tone carries information that a transcript loses, which matters for call-centre analytics, voice assistants and accessibility tools. This project is a compact end-to-end example: audio preprocessing, two models compared on the same split (a random-forest baseline on MFCC statistics and a CNN on spectrograms), and a demo that runs the trained CNN.

## Results (from the training notebook)

4 of RAVDESS's 8 emotions, 672 clips from 24 actors, stratified 80/20 split (135 test clips):

| Model | Input | Test accuracy | Macro F1 |
|---|---|---|---|
| Random forest (200 trees) | 40 MFCC means + 40 standard deviations | 71.1% | 0.69 |
| CNN (3 conv blocks, 1.07 M parameters) | 128 × 128 log-mel spectrogram | 68.1% | 0.65 |

The CNN is the model in the demo (`best_mel_cnn.h5`), saved at its best validation accuracy. On this small dataset the simpler baseline scored slightly higher. Both are weakest on *neutral* (recall 0.37 to 0.47), which has half as many clips as the other classes.

## How it works

```mermaid
flowchart LR
    A[Audio clip] --> B[Load 16 kHz mono<br/>trim silence, top_db=20<br/>peak-normalise]
    B --> C[128-band log-mel<br/>n_fft 2048, hop 512]
    C --> D[Centre pad or crop<br/>to 128 frames]
    D --> E[CNN<br/>3 × Conv-BN-Pool<br/>Dense + Dropout]
    E --> F[Softmax over<br/>angry · happy · neutral · sad]
```

`emotion.py` holds the inference code, written to match the notebook's preprocessing step for step; `app.py` is the Gradio demo and `streamlit_app.py` a Streamlit version of it.

## Run it

```bash
pip install -r requirements.txt
python app.py                    # Gradio demo on http://127.0.0.1:7860
# or
streamlit run streamlit_app.py

pip install pytest
python -m pytest tests           # 7 checks on the inference code (about 5 s)
```

To retrain, open `ravdess_training.ipynb`; its first cell downloads RAVDESS (about 215 MB) from Zenodo.

## Fixed on 2026-11-13

- **The demo reported the wrong emotion.** The Streamlit app mapped outputs as `{0: neutral, 1: happy, 2: sad, 3: angry}`, but the notebook's `LabelEncoder` sorts classes alphabetically, `{0: angry, 1: happy, 2: neutral, 3: sad}`. A model output of *angry* was shown as *neutral*, *neutral* as *sad* and *sad* as *angry*. `emotion.py` now uses the training order.
- **Preprocessing did not match training.** The app trimmed silence with librosa's default `top_db=60`; training used `top_db=20`. It now uses 20.
- Each upload is written to its own temporary file (all users shared `temp.wav`), and the model is loaded once instead of on every prediction.

## Fixed on 2026-11-25

- **A silent recording was still classified.** `librosa.effects.trim` keeps an all-silent clip whole, and peak-normalising near-silence turns faint hiss into a full-scale signal, so two seconds of silence came back as an emotion. Clips quieter than about -60 dBFS now get "The clip is too short or silent". Tests in `tests/` cover it.

## Known limitations

- **Speaker overlap.** The split is by clip, not by actor, so test clips come from actors the model heard in training. Accuracy on new speakers is likely lower; an actor-held-out split would measure it.
- **Small, acted data.** 672 clips of two sentences read by 24 North American actors. Spontaneous speech, other accents, noise and phone microphones are outside what it saw.
- **No labelled-audio test.** The notebook records the evaluation. `tests/` checks the inference code without labelled audio: the label order matches the notebook's `LabelEncoder`, the preprocessing gives the training shape for short and long clips, silence is refused, and a prediction is one probability per label summing to 1.
- **4 classes only.** Calm, fearful, disgust and surprised were dropped to keep the task balanced and small.

## Deploy to Hugging Face Spaces

Free, CPU-only. In a browser:

1. Sign in at [huggingface.co](https://huggingface.co) and choose **New → Space**. Name it `speech-emotion-classifier`, SDK **Gradio**, hardware **CPU basic (free)**, visibility **Public**.
2. In the Space's **Files** tab choose **Add file → Upload files** and upload: `app.py`, `emotion.py`, `best_mel_cnn.h5`, and from `space/` both `README.md` and `requirements.txt` (they replace the Space's defaults).
3. Wait for the build (a few minutes for TensorFlow). The Space is then live at `https://huggingface.co/spaces/<your-username>/speech-emotion-classifier`.
4. Put that link in the "Try it" line at the top of this README.

## Data and licence

The model is trained on the speech part of RAVDESS: Livingstone SR, Russo FA (2018), *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLoS ONE 13(5): e0196391, [doi:10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391). RAVDESS is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), so the trained weights (`best_mel_cnn.h5`) are for non-commercial use with attribution.

---

Built by [Nityashree S](https://nityashree.in) · [GitHub](https://github.com/Nityaa08)
