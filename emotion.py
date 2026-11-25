"""
Inference for the speech-emotion CNN trained in `ravdess_training.ipynb`.

The preprocessing here must match the notebook exactly, or the model sees
different inputs from the ones it was trained on:
  - load at 16 kHz mono, trim silence with top_db=20, peak-normalise
  - 128-band log-mel spectrogram (n_fft=2048, hop_length=512), ref=max
  - centre-pad (with the spectrogram's minimum) or centre-crop to 128 frames

LABELS follows the notebook's LabelEncoder, which sorts class names
alphabetically: {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}.
"""
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

MODEL_PATH = Path(__file__).with_name("best_mel_cnn.h5")
SAMPLE_RATE = 16_000
N_MELS = 128
FRAMES = 128
LABELS = ["angry", "happy", "neutral", "sad"]  # LabelEncoder order from the notebook


# Below about -60 dBFS there is nothing to classify. Without this, trim() keeps an
# all-silent clip whole, and peak-normalising near-silence turns hiss into a full-scale
# signal, so a silent recording still came back as an emotion.
SILENCE_PEAK = 1e-3


def load_audio(path: str) -> np.ndarray:
    """The clip as the model was trained on it; empty if it is silent."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak < SILENCE_PEAK:
        return np.zeros(0, dtype=np.float32)
    return y / peak


def mel_features(y: np.ndarray) -> np.ndarray:
    s = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS)
    s_db = librosa.power_to_db(s, ref=np.max)
    t = s_db.shape[1]
    if t < FRAMES:
        left = (FRAMES - t) // 2
        s_db = np.pad(s_db, ((0, 0), (left, FRAMES - t - left)), mode="constant", constant_values=s_db.min())
    elif t > FRAMES:
        start = (t - FRAMES) // 2
        s_db = s_db[:, start:start + FRAMES]
    return s_db[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1, 128, 128, 1)


@lru_cache(maxsize=1)
def model():
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH, compile=False)


def predict(path: str) -> dict[str, float]:
    """Class probabilities for one audio file, e.g. {'angry': 0.71, ...}."""
    y = load_audio(path)
    if y.size < SAMPLE_RATE // 10:
        raise ValueError("The clip is too short or silent. Record at least one spoken sentence.")
    probs = model().predict(mel_features(y), verbose=0)[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}
