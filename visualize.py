# src/visualize.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from .train_from_lab import collect_pairs, parse_lab, SAMPLE_RATE, CLIP_SECONDS, WINDOW_STRIDE, N_MELS, feature_vector
from .features import load_mono, clip_or_pad, extract_logmel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

# --- helper: build a small test set (re-create same logic as training but can be limited) ---
def build_test_pairs(limit_files=30, max_segments_per_file=100):
    pairs = collect_pairs(ROOT / "kaggleData" / "archive")
    selected = pairs[:limit_files]
    X = []
    y = []
    sample_meta = []  # keep (wav, t0, t1)
    for wav_path, lab_path in selected:
        try:
            y_audio, sr = load_mono(str(wav_path), sr=SAMPLE_RATE)
        except Exception:
            continue
        intervals = parse_lab(lab_path)
        for (start, end, label) in intervals:
            if label not in ("bee", "nobee"): 
                continue
            if end - start < CLIP_SECONDS:
                continue
            seg_count = 0
            t = start
            while t + CLIP_SECONDS <= end and seg_count < max_segments_per_file:
                s0 = int(t * sr); s1 = int((t + CLIP_SECONDS) * sr)
                seg = y_audio[s0:s1]
                if len(seg) < int(CLIP_SECONDS * sr):
                    seg = clip_or_pad(seg, sr, CLIP_SECONDS)
                logmel = extract_logmel(seg, sr, n_mels=N_MELS)
                vec = feature_vector(logmel)
                X.append(vec)
                y.append(label)
                sample_meta.append((wav_path.name, t, t+CLIP_SECONDS))
                seg_count += 1
                t += WINDOW_STRIDE
    return np.array(X), np.array(y), sample_meta

def plot_confusion(cm, labels):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

def plot_class_counts(y):
    uniq, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(5,4))
    sns.barplot(x=list(uniq), y=list(counts))
    plt.title("Class counts (sampled test set)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_counts.png", dpi=200)
    plt.close()

def plot_spectrogram_example(wav_name, t0, t1, base_dir):
    wav_path = next((p for p in base_dir.rglob(wav_name)), None)
    if not wav_path:
        return
    y_audio, sr = load_mono(str(wav_path), sr=SAMPLE_RATE)
    s0 = int(t0*sr); s1 = int(t1*sr)
    seg = y_audio[s0:s1]
    if len(seg) < int(CLIP_SECONDS*sr):
        seg = clip_or_pad(seg, sr, CLIP_SECONDS)
    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=int(sr*0.01), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel spectrogram: {wav_name} @ {t0:.1f}s")
    plt.tight_layout()
    outname = OUT_DIR / f"spectrogram_{wav_name}_{int(t0)}.png"
    plt.savefig(outname, dpi=200)
    plt.close()

def plot_roc_pr(clf, X, y, labels):
    # Binarize labels for ROC/PR
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    classes = lb.classes_
    if Y.shape[1] == 1:
        # binary but LabelBinarizer gives single column; make two-column
        Y = np.hstack([1-Y, Y])
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
    else:
        # try decision_function
        try:
            scores = clf.decision_function(X)
            proba = np.vstack([1-scores, scores]).T
        except Exception:
            print("Model lacks predict_proba / decision_function; skipping ROC/PR")
            return

    # ROC per class
    plt.figure(figsize=(6,5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=200)
    plt.close()

    # PR curve
    plt.figure(figsize=(6,5))
    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(Y[:, i], proba[:, i])
        ap = auc(rec, prec)  # area under PR curve
        plt.plot(rec, prec, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves.png", dpi=200)
    plt.close()

def main():
    print("Building a sampled test set (may take a minute)...")
    X, y, meta = build_test_pairs(limit_files=40, max_segments_per_file=50)
    if X.size == 0:
        print("No test segments found.")
        return

    # load model (pick first joblib in models/)
    model_files = list(MODEL_DIR.glob("*.joblib"))
    if not model_files:
        print("No model joblib found in", MODEL_DIR)
        return
    clf = load(model_files[0])
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    plot_confusion(cm, labels=np.unique(y))
    plot_class_counts(y)
    plot_roc_pr(clf, X, y, labels=np.unique(y))

    # save some spectrogram examples: pick a few true positives and false negatives
    # find indices of false negatives (true==bee, pred==nobee)
    fp_idx = [i for i,(t,p) in enumerate(zip(y, y_pred)) if t!=p]
    example_indices = (fp_idx[:3] + list(range(3)))[:6]
    base_dir = ROOT / "kaggleData" / "archive"
    for idx in example_indices:
        wav_name, t0, t1 = meta[idx]
        plot_spectrogram_example(wav_name, t0, t1, base_dir)
    print("Saved plots to", OUT_DIR)

if __name__ == "__main__":
    main()
