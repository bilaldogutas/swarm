# src/features.py
import librosa
import numpy as np

def extract_logmel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    frame_length: float = 0.025,   # 25 ms
    hop_length: float = 0.010,     # 10 ms
    fmin: int = 20,
    fmax: int | None = None
) -> np.ndarray:
    """Return (T, n_mels) log-mel spectrogram."""
    n_fft = int(sr * frame_length)
    hop = int(sr * hop_length)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    logS = librosa.power_to_db(S, ref=np.max)
    return logS.T  # time-major

def load_mono(filepath: str, sr: int = 48000, duration: float | None = None) -> tuple[np.ndarray, int]:
    """Load audio as mono with a target sample rate (default 48k)."""
    y, sr = librosa.load(filepath, sr=sr, mono=True, duration=duration)
    # simple amplitude guard
    if np.abs(y).max() > 1.0:
        y = y / np.abs(y).max()
    return y, sr

def clip_or_pad(y: np.ndarray, sr: int, target_seconds: float = 2.0) -> np.ndarray:
    """Force a fixed-length clip for consistent feature shapes."""
    target_len = int(target_seconds * sr)
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        pad = target_len - len(y)
        return np.pad(y, (0, pad))
    return y
