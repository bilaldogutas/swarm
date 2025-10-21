# src/train_from_lab.py
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from .features import load_mono, extract_logmel, clip_or_pad

# 🔧 your dataset root (contains wav + lab pairs)
DATA_DIR = Path(r"C:\Users\JegMcIntire\Documents\bee_project\kaggleData\archive")
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE   = 48000
CLIP_SECONDS  = 2.0              # window length for training examples
WINDOW_STRIDE = 1.0              # slide stride in seconds (controls how many windows you get)
N_MELS        = 64
CLASSIFIER    = "logreg"         # or "rf" for RandomForest

def parse_lab(lab_path: Path):
    """
    Parse a .lab file:
      first line: identifier (ignore for alignment)
      subsequent lines: start_s end_s label
    returns list of (start, end, label)
    """
    intervals = []
    with open(lab_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return intervals
    for ln in lines[1:]:  # skip first identifier line
        parts = ln.split()
        if len(parts) < 3:
            continue
        start, end = float(parts[0]), float(parts[1])
        label = parts[2].lower()   # expect 'bee' or 'nobee'
        intervals.append((start, end, label))
    return intervals

def windows_from_interval(start, end, win, stride):
    """Yield [t0, t1) windows inside [start, end]."""
    t = start
    while t + win <= end + 1e-6:
        yield (t, t + win)
        t += stride

def feature_vector(logmel_TxM: np.ndarray) -> np.ndarray:
    # simple stats pooling (mean+std) → fixed-size vector
    mu = logmel_TxM.mean(axis=0)
    sd = logmel_TxM.std(axis=0)
    return np.concatenate([mu, sd], axis=0)

def collect_pairs(root: Path):
    """Find all wav files and their matching .lab files."""
    wavs = glob.glob(str(root / "**" / "*.wav"), recursive=True)
    pairs = []
    for w in wavs:
        wpath = Path(w)
        lab = wpath.with_suffix(".lab")
        if lab.exists():
            pairs.append((wpath, lab))
    return pairs

def main():
    pairs = collect_pairs(DATA_DIR)
    if not pairs:
        print(f"🙈 No wav+lab pairs found under {DATA_DIR}")
        return

    X, y = [], []
    for wav_path, lab_path in tqdm(pairs, desc="Processing files"):
        # load full audio once
        try:
            y_audio, sr = load_mono(str(wav_path), sr=SAMPLE_RATE)
        except Exception as e:
            print(f"⚠️ Could not load {wav_path}: {e}")
            continue

        # parse label intervals
        intervals = parse_lab(lab_path)
        if not intervals:
            continue

        dur = len(y_audio) / sr
        for (start, end, label) in intervals:
            # clamp to file duration
            start = max(0.0, start)
            end   = min(dur, end)
            if end - start < CLIP_SECONDS:
                continue
            if label not in ("bee", "nobee"):
                continue  # ignore other labels if present

            for t0, t1 in windows_from_interval(start, end, CLIP_SECONDS, WINDOW_STRIDE):
                s0 = int(t0 * sr); s1 = int(t1 * sr)
                seg = y_audio[s0:s1]
                if len(seg) < int(CLIP_SECONDS * sr):
                    seg = clip_or_pad(seg, sr, CLIP_SECONDS)

                logmel = extract_logmel(seg, sr, n_mels=N_MELS)
                vec = feature_vector(logmel)
                X.append(vec)
                y.append(label)

    if not X:
        print("No training segments were created. Check labels and durations.")
        return

    X = np.array(X)
    y = np.array(y)
    print("Dataset segments:", X.shape[0], " Feature dim:", X.shape[1])
    print("Classes:", {c: (y == c).sum() for c in np.unique(y)})

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if CLASSIFIER == "logreg":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300))
        ])
        model_name = "bee_nobee_logmel_logreg.joblib"
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
        model_name = "bee_nobee_logmel_rf.joblib"

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("\n=== Classification Report ===")
    print(classification_report(y_te, y_pred, digits=4))
    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(confusion_matrix(y_te, y_pred),
                       index=np.unique(y), columns=np.unique(y)))

    dump(clf, MODEL_DIR / model_name)
    with open(MODEL_DIR / "model_meta.txt", "w") as f:
        f.write(f"classifier={CLASSIFIER}\n")
        f.write(f"sr={SAMPLE_RATE}\n")
        f.write(f"clip_seconds={CLIP_SECONDS}\n")
        f.write(f"n_mels={N_MELS}\n")
        f.write(f"stride={WINDOW_STRIDE}\n")
    print(f"\n✅ Saved model to {MODEL_DIR / model_name}")

if __name__ == "__main__":
    main()
