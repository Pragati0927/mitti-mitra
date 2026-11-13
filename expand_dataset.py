"""
Mitti Mitra - Dataset Expansion Script
Generates a large, realistic synthetic dataset for Indian (especially Indo-Gangetic / UP)
crop rotation recommendations.

Supports Phase 1 inputs:
- Location coordinates (lat/lon)
- Weather (temperature, humidity, rainfall)
- Past crop → used to infer soil nutrient status (N, P, K, pH)

Output: master_indian_recommendation_dataset_expanded.csv
         (ready for training)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ---------------------------------------------------------------------------
# Domain knowledge tables
# ---------------------------------------------------------------------------
CROPS = [
    "Wheat", "Rice", "Sugarcane", "Potato", "Mustard",
    "Maize", "Gram", "Moong", "Arhar", "Soybean", "Fallow"
]

# Baseline soil for alluvial Indo-Gangetic plain (Greater Noida / West UP)
BASELINE_SOIL = {
    "N_ppm": 120,
    "P_ppm": 25,
    "K_ppm": 220,
    "pH": 7.4,
    "OM_pct": 0.8,
}

# Nutrient depletion / residual effect of previous crop
# Positive = depletion (lower residual), Negative = residual benefit
NUTRIENT_EFFECT = {
    "Wheat":     {"N": -40, "P": -12, "K": -25, "pH": -0.1},
    "Rice":      {"N": -50, "P": -10, "K": -20, "pH": -0.15},
    "Sugarcane": {"N": -70, "P": -20, "K": -40, "pH": -0.2},
    "Potato":    {"N": -45, "P": -15, "K": -60, "pH": -0.1},
    "Mustard":   {"N": -25, "P": -8,  "K": -15, "pH": 0.0},
    "Maize":     {"N": -55, "P": -15, "K": -30, "pH": -0.1},
    "Gram":      {"N": +30, "P": -10, "K": -10, "pH": 0.05},   # N-fixing
    "Moong":     {"N": +25, "P": -8,  "K": -8,  "pH": 0.05},
    "Arhar":     {"N": +20, "P": -12, "K": -12, "pH": 0.0},
    "Soybean":   {"N": +35, "P": -12, "K": -15, "pH": 0.05},
    "Fallow":    {"N": +5,  "P": +2,  "K": +3,  "pH": 0.0},
}

# Preferred next crop given past crop (rotation knowledge) + probability weights
ROTATION_PREFERENCES = {
    "Wheat":     [("Moong", 0.25), ("Gram", 0.20), ("Mustard", 0.15), ("Maize", 0.15), ("Rice", 0.10), ("Fallow", 0.05), ("Potato", 0.05), ("Soybean", 0.05)],
    "Rice":      [("Wheat", 0.30), ("Mustard", 0.20), ("Moong", 0.15), ("Potato", 0.10), ("Maize", 0.10), ("Gram", 0.10), ("Fallow", 0.05)],
    "Sugarcane": [("Wheat", 0.25), ("Mustard", 0.20), ("Moong", 0.15), ("Maize", 0.15), ("Gram", 0.15), ("Fallow", 0.10)],
    "Potato":    [("Wheat", 0.25), ("Moong", 0.20), ("Mustard", 0.15), ("Maize", 0.15), ("Gram", 0.15), ("Rice", 0.10)],
    "Mustard":   [("Moong", 0.20), ("Maize", 0.20), ("Rice", 0.15), ("Wheat", 0.15), ("Soybean", 0.15), ("Gram", 0.15)],
    "Maize":     [("Wheat", 0.25), ("Mustard", 0.20), ("Moong", 0.15), ("Gram", 0.15), ("Rice", 0.15), ("Soybean", 0.10)],
    "Gram":      [("Rice", 0.25), ("Maize", 0.20), ("Wheat", 0.15), ("Mustard", 0.15), ("Potato", 0.15), ("Sugarcane", 0.10)],
    "Moong":     [("Rice", 0.25), ("Wheat", 0.20), ("Maize", 0.20), ("Mustard", 0.15), ("Potato", 0.10), ("Sugarcane", 0.10)],
    "Arhar":     [("Wheat", 0.25), ("Mustard", 0.20), ("Maize", 0.20), ("Rice", 0.15), ("Potato", 0.10), ("Moong", 0.10)],
    "Soybean":   [("Wheat", 0.30), ("Mustard", 0.20), ("Maize", 0.20), ("Rice", 0.15), ("Gram", 0.15)],
    "Fallow":    [("Wheat", 0.25), ("Rice", 0.20), ("Maize", 0.15), ("Mustard", 0.15), ("Potato", 0.15), ("Sugarcane", 0.10)],
}

# Growing requirements (simple, returned at inference time)
GROWING_REQUIREMENTS = {
    "Wheat":     "Rabi crop. Sow Nov. Needs 4-5 irrigations. Prefers cool temp (15-25°C).",
    "Rice":      "Kharif crop. Sow June-July. High water need. Warm temp (>25°C).",
    "Sugarcane": "Long duration. Plant Feb-Mar or Oct. High water + NPK demand.",
    "Potato":    "Rabi/Zaid. Cool climate. High K demand. Well-drained soil.",
    "Mustard":   "Rabi. Sow Oct-Nov. Low water need after establishment. Cool season.",
    "Maize":     "Kharif or Zaid. Fast growing. Responsive to N. Avoid waterlogging.",
    "Gram":      "Rabi pulse. N-fixing. Low water. Sow Oct-Nov. Improves soil.",
    "Moong":     "Short duration (60-70 days). Kharif/Zaid/Summer. N-fixing. Good for rotation.",
    "Arhar":     "Long duration pulse. Kharif. N-fixing. Drought tolerant once established.",
    "Soybean":   "Kharif. N-fixing. Moderate water. Good preceding crop for wheat.",
    "Fallow":    "Rest period. Helps recover soil moisture and nutrients.",
}

# Approximate coordinate ranges for major agro-zones (biased toward West UP / NCR)
LOCATION_ZONES = [
    {"name": "West UP / NCR", "lat": (28.3, 28.8), "lon": (77.3, 77.7), "weight": 0.45},
    {"name": "Central UP",    "lat": (26.5, 27.5), "lon": (80.5, 81.5), "weight": 0.20},
    {"name": "East UP",       "lat": (25.5, 26.5), "lon": (82.5, 83.5), "weight": 0.15},
    {"name": "Punjab / Haryana", "lat": (30.0, 31.0), "lon": (75.0, 77.0), "weight": 0.12},
    {"name": "Bihar",         "lat": (25.0, 26.0), "lon": (84.5, 85.5), "weight": 0.08},
]


def sample_location():
    zone = np.random.choice(LOCATION_ZONES, p=[z["weight"] for z in LOCATION_ZONES])
    lat = np.random.uniform(*zone["lat"])
    lon = np.random.uniform(*zone["lon"])
    return lat, lon, zone["name"]


def infer_soil_from_past_crop(past_crop: str) -> dict:
    """Apply nutrient effect of past crop on baseline soil."""
    effect = NUTRIENT_EFFECT.get(past_crop, {"N": 0, "P": 0, "K": 0, "pH": 0})
    soil = {
        "N_ppm": max(20, BASELINE_SOIL["N_ppm"] + effect["N"] + np.random.normal(0, 8)),
        "P_ppm": max(5,  BASELINE_SOIL["P_ppm"] + effect["P"] + np.random.normal(0, 3)),
        "K_ppm": max(50, BASELINE_SOIL["K_ppm"] + effect["K"] + np.random.normal(0, 15)),
        "pH":    np.clip(BASELINE_SOIL["pH"] + effect["pH"] + np.random.normal(0, 0.15), 5.5, 8.5),
        "OM_pct": max(0.3, BASELINE_SOIL["OM_pct"] + np.random.normal(0, 0.15)),
    }
    return soil


def sample_weather(lat: float) -> dict:
    """Rough seasonal weather for North India (biased to current typical conditions)."""
    # Simple model: higher latitude slightly cooler
    base_temp = 28 - (lat - 28) * 1.5
    temp = np.random.normal(base_temp, 4)
    humidity = np.random.uniform(40, 85)
    # Rainfall: mostly low outside monsoon; we mix
    if np.random.rand() < 0.3:          # monsoon-like
        rainfall = np.random.uniform(20, 120)
    else:
        rainfall = np.random.uniform(0, 15)
    return {
        "temperature_c": round(temp, 1),
        "humidity_pct": round(humidity, 1),
        "rainfall_mm": round(rainfall, 1),
    }


def generate_sample() -> dict:
    past_crop = np.random.choice(CROPS)
    lat, lon, zone = sample_location()
    soil = infer_soil_from_past_crop(past_crop)
    weather = sample_weather(lat)

    # Choose recommended crop according to rotation preference (with controlled noise)
    prefs = ROTATION_PREFERENCES.get(past_crop, [(c, 1/len(CROPS)) for c in CROPS])
    crops, weights = zip(*prefs)
    # Stronger signal: mostly follow preference, small noise
    weights = np.array(weights) * 0.85 + np.random.uniform(0, 0.15, size=len(weights))
    weights /= weights.sum()
    recommended = np.random.choice(crops, p=weights)

    return {
        "past_crop": past_crop,
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "zone": zone,
        "temperature_c": weather["temperature_c"],
        "humidity_pct": weather["humidity_pct"],
        "rainfall_mm": weather["rainfall_mm"],
        "inferred_N_ppm": round(soil["N_ppm"], 1),
        "inferred_P_ppm": round(soil["P_ppm"], 1),
        "inferred_K_ppm": round(soil["K_ppm"], 1),
        "inferred_pH": round(soil["pH"], 2),
        "inferred_OM_pct": round(soil["OM_pct"], 2),
        "recommended_crop": recommended,
    }


def main(n_samples: int = 15000, output_path: str = "master_indian_recommendation_dataset_expanded.csv"):
    print(f"Generating {n_samples} realistic samples...")
    records = [generate_sample() for _ in range(n_samples)]
    df = pd.DataFrame(records)

    # Basic quality checks
    print(f"Dataset shape: {df.shape}")
    print(f"Unique past crops: {df['past_crop'].nunique()}")
    print(f"Unique recommended crops: {df['recommended_crop'].nunique()}")
    print(f"Zones: {df['zone'].value_counts().to_dict()}")

    out = Path(output_path)
    df.to_csv(out, index=False)
    print(f"Saved → {out.resolve()}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    main()
