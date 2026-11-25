"""Checks on the inference code: label order, preprocessing shape, and a prediction.

No labelled audio is needed: these guard against the bugs fixed on 2026-11-13 (a
wrong label order and preprocessing that did not match training) coming back.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import emotion

ROOT = Path(__file__).resolve().parent.parent


def tone(path: Path, seconds: float = 3.0, sr: int = 16_000) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)), sr)
    return path


def test_labels_follow_the_notebooks_label_encoder():
    # LabelEncoder sorts the class names; the demo once mapped 0 to "neutral".
    nb = json.loads((ROOT / "ravdess_training.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "LabelEncoder" in code
    assert emotion.LABELS == sorted(["neutral", "happy", "sad", "angry"])


def test_preprocessing_matches_training(tmp_path):
    y = emotion.load_audio(str(tone(tmp_path / "a.wav")))
    assert abs(float(np.max(np.abs(y))) - 1.0) < 1e-6  # peak-normalised
    x = emotion.mel_features(y)
    assert x.shape == (1, emotion.N_MELS, emotion.FRAMES, 1) and x.dtype == np.float32


@pytest.mark.parametrize("seconds", [0.5, 10.0])
def test_short_and_long_clips_are_padded_or_cropped(tmp_path, seconds):
    x = emotion.mel_features(emotion.load_audio(str(tone(tmp_path / "b.wav", seconds))))
    assert x.shape == (1, 128, 128, 1)


@pytest.mark.parametrize("signal", [
    np.zeros(16_000 * 2),                                        # digital silence
    np.random.default_rng(0).normal(0, 1e-4, 16_000 * 2),        # a quiet room
])
def test_a_silent_clip_is_refused(tmp_path, signal):
    # Before 2026-11-25 two seconds of silence were classified (trim keeps an all-silent
    # clip whole, and peak-normalising turned faint hiss into a full-scale signal).
    path = tmp_path / "silent.wav"
    sf.write(path, signal, 16_000)
    with pytest.raises(ValueError, match="too short or silent"):
        emotion.predict(str(path))


def test_prediction_is_a_probability_for_each_label(tmp_path):
    pytest.importorskip("tensorflow")
    probs = emotion.predict(str(tone(tmp_path / "c.wav")))
    assert list(probs) == emotion.LABELS
    assert abs(sum(probs.values()) - 1.0) < 1e-4
