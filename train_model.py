# train_model.py

"""
This script trains a TINY fake ML model on FAKE data,
using your real feature extraction functions.

Later, you will replace the fake data with real hive FFT features.
"""

import random
import joblib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# Import YOUR feature extraction functions
from processing.features import (
    mean,
    maximum,
    minimum,
    energy,
    std_dev,
    low_high_band_energy,
)


def extract_features(fft_values):
    """Turn fft_values into a feature vector for the ML model."""
    abs_vals = [abs(v) for v in fft_values]

    avg = mean(abs_vals)
    max_val = maximum(abs_vals)
    min_val = minimum(abs_vals)
    en = energy(abs_vals)
    sd = std_dev(abs_vals)
    low_en, high_en = low_high_band_energy(abs_vals)

    # Return as a list/array for the model
    return [avg, max_val, min_val, en, sd, low_en, high_en]


def generate_fake_training_data(num_samples=200):
    """
    Create fake FFT samples with fake labels.

    We *explicitly* create:
      - about half samples with low high-band energy (label 0)
      - about half samples with high high-band energy (label 1)

    This guarantees we have BOTH classes for training.
    """

    X = []
    y = []

    for i in range(num_samples):
        if i < num_samples // 2:
            # Class 0: "stable" hive
            #   - higher energy in LOW band
            #   - low energy in HIGH band
            low_part = [random.uniform(0.5, 1.0) for _ in range(16)]
            high_part = [random.uniform(0.0, 0.2) for _ in range(16)]
            label = 0
        else:
            # Class 1: "swarm risk"
            #   - low energy in LOW band
            #   - high energy in HIGH band
            low_part = [random.uniform(0.0, 0.2) for _ in range(16)]
            high_part = [random.uniform(0.5, 1.0) for _ in range(16)]
            label = 1

        fft_vals = low_part + high_part

        # Extract features
        feats = extract_features(fft_vals)
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)



def main():
    print("Generating fake training data...")
    X, y = generate_fake_training_data()

    print("Training logistic regression model...")
    model = LogisticRegression()
    model.fit(X, y)

    # Save the model
    model_path = Path("models/swarm_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path.absolute()}")


if __name__ == "__main__":
    main()
