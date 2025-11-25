from datetime import datetime
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from processing.risk import simple_risk_from_fft, label_from_risk

from processing.features import (
    mean,
    maximum,
    minimum,
    energy,
    std_dev,
    low_high_band_energy,
)


# 1. Create the FastAPI app instance
app = FastAPI(
    title="HoneyHz Backend",
    description="Backend API for hive swarm prediction",
    version="0.1.0"
)

class FeatureDebugResponse(BaseModel):
    hive_id: int
    timestamp: datetime
    mean_value: float
    max_value: float
    min_value: float
    energy_value: float
    std_dev_value: float
    low_band_energy: float
    high_band_energy: float




# 2. Define the shape of the data the backend expects
class PredictionRequest(BaseModel):
    hive_id: int
    timestamp: datetime
    # Later this might be raw vibration or FFT features;
    # for now we just accept a list of numbers.
    fft_values: List[float]


class PredictionResponse(BaseModel):
    hive_id: int
    timestamp: datetime
    swarm_risk: float          # 0.0 to 1.0
    risk_level: str            # "Stable", "Watch", "High Risk"


# 3. Health-check endpoint (for "are you alive?" checks)
@app.get("/health")
def health_check():
    return {"status": "ok"}


# 4. Prediction endpoint (currently uses a FAKE model)
@app.post("/predict", response_model=PredictionResponse)
def predict_swarm(request: PredictionRequest):
    """
    Use a VERY simple rule to turn fft_values into a swarm_risk.
    Later this will be replaced by real signal processing + a trained ML model.
    """

    # 1. Compute a risk number based on the FFT values
    swarm_risk = simple_risk_from_fft(request.fft_values)

    # 2. Turn that number into a text label
    if swarm_risk < 0.4:
        risk_level = "Stable"
    elif swarm_risk < 0.7:
        risk_level = "Watch"
    else:
        risk_level = "High Risk"

    # 3. Build the response object
    return PredictionResponse(
        hive_id=request.hive_id,
        timestamp=request.timestamp,
        swarm_risk=swarm_risk,
        risk_level=risk_level,
    )


@app.post("/debug/features", response_model=FeatureDebugResponse)
def debug_features(request: PredictionRequest):
    """
    Debug endpoint:
    Given the same input as /predict, return the basic features
    we compute from fft_values. This helps you understand what
    the backend "sees" before making a risk prediction.
    """

    values = request.fft_values

    mean_val = mean(values)
    max_val = maximum(values)
    min_val = minimum(values)
    energy_val = energy(values)
    std_val = std_dev(values)
    low_energy, high_energy = low_high_band_energy(values)

    return FeatureDebugResponse(
        hive_id=request.hive_id,
        timestamp=request.timestamp,
        mean_value=mean_val,
        max_value=max_val,
        min_value=min_val,
        energy_value=energy_val,
        std_dev_value=std_val,
        low_band_energy=low_energy,
        high_band_energy=high_energy,
    )

