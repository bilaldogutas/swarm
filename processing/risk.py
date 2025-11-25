# processing/risk.py

from typing import List

from processing.features import (
    mean,
    maximum,
    energy,
    low_high_band_energy,
)



def simple_risk_from_fft(fft_values: List[float]) -> float:
    """
    Placeholder risk logic that uses a few basic features
    of the FFT values instead of raw numbers.

    NEW: incorporates separate energy in "low" and "high" bands.
    For now we pretend the high band matters more (like queen piping).
    """

    if not fft_values:
        return 0.1  # tiny risk if we have no data

    # 1. Compute some features
    abs_values = [abs(v) for v in fft_values]
    avg_mag = mean(abs_values)
    max_mag = maximum(abs_values)
    total_energy = energy(fft_values)
    low_energy, high_energy = low_high_band_energy(fft_values)

    # 2. Combine them into a single score.
    #    We weight the high-band energy more heavily to mimic the idea
    #    that high-frequency queen piping (e.g., 200–500 Hz) is important.
    raw_score = (
        0.3 * avg_mag +
        0.2 * max_mag +
        0.1 * (total_energy / 10.0) +
        0.15 * (low_energy / 10.0) +
        0.25 * (high_energy / 10.0)
    )

    # 3. Clamp to [0.0, 1.0]
    if raw_score < 0.0:
        raw_score = 0.0
    if raw_score > 1.0:
        raw_score = 1.0

    return raw_score




def label_from_risk(swarm_risk: float) -> str:
    """
    Convert a numeric risk into a text label
    that the app will display.
    """

    if swarm_risk < 0.4:
        return "Stable"
    elif swarm_risk < 0.7:
        return "Watch"
    else:
        return "High Risk"
