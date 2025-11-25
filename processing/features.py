# processing/features.py

from typing import List
import math


def mean(values: List[float]) -> float:
    """
    Compute the average (mean) of a list of numbers.
    If the list is empty, return 0.0 to avoid crashes.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def maximum(values: List[float]) -> float:
    """
    Largest value in the list.
    If the list is empty, return 0.0.
    """
    if not values:
        return 0.0
    return max(values)


def minimum(values: List[float]) -> float:
    """
    Smallest value in the list.
    If the list is empty, return 0.0.
    """
    if not values:
        return 0.0
    return min(values)


def energy(values: List[float]) -> float:
    """
    Very simple "signal energy":
    sum of squares of the values.

    In signal processing, energy ~ how strong the signal is overall.
    """
    if not values:
        return 0.0
    return sum(v * v for v in values)


def std_dev(values: List[float]) -> float:
    """
    Standard deviation: how spread out the values are.
    High std_dev means the signal is very "spiky".
    """
    if not values:
        return 0.0

    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def band_energy(values: List[float], start_idx: int, end_idx: int) -> float:
    """
    Compute the "energy" (sum of squares) in a slice of the FFT array:
    values[start_idx:end_idx]

    If the indices are out of range or the slice is empty, return 0.0.
    """
    n = len(values)
    if n == 0:
        return 0.0

    # Clamp indices to valid range
    start = max(0, start_idx)
    end = min(n, end_idx)

    if start >= end:
        return 0.0

    return sum(v * v for v in values[start:end])


def low_high_band_energy(values: List[float]) -> tuple[float, float]:
    """
    Very simple placeholder:
    - Treat the first half of the list as "low frequencies"
    - Treat the second half as "high frequencies"

    Returns (low_band_energy, high_band_energy).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0

    mid = n // 2  # integer division
    low = band_energy(values, 0, mid)
    high = band_energy(values, mid, n)
    return low, high
