"""
Mitti Mitra - Updated Prediction Engine (Phase 1 compatible)
Accepts:
  - past_crop
  - latitude, longitude  (or state/district for fallback)
  - temperature_c, humidity_pct, rainfall_mm
  - optional field_size

Returns:
  - top 2-4 recommended crops with confidence
  - inferred missing / depleted soil nutrients
  - basic growing requirements for each recommendation
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Domain knowledge (must stay in sync with expand_dataset.py)
# ---------------------------------------------------------------------------
NUTRIENT_EFFECT = {
    "Wheat":     {"N": -40, "P": -12, "K": -25, "pH": -0.1},
    "Rice":      {"N": -50, "P": -10, "K": -20, "pH": -0.15},
    "Sugarcane": {"N": -70, "P": -20, "K": -40, "pH": -0.2},
    "Potato":    {"N": -45, "P": -15, "K": -60, "pH": -0.1},
    "Mustard":   {"N": -25, "P": -8,  "K": -15, "pH": 0.0},
    "Maize":     {"N": -55, "P": -15, "K": -30, "pH": -0.1},
    "Gram":      {"N": +30, "P": -10, "K": -10, "pH": 0.05},
    "Moong":     {"N": +25, "P": -8,  "K": -8,  "pH": 0.05},
    "Arhar":     {"N": +20, "P": -12, "K": -12, "pH": 0.0},
    "Soybean":   {"N": +35, "P": -12, "K": -15, "pH": 0.05},
    "Fallow":    {"N": +5,  "P": +2,  "K": +3,  "pH": 0.0},
}

BASELINE_SOIL = {"N_ppm": 120, "P_ppm": 25, "K_ppm": 220, "pH": 7.4, "OM_pct": 0.8}

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


def infer_soil_nutrients(past_crop: str) -> Dict[str, float]:
    """Return current estimated soil nutrients after the past crop."""
    effect = NUTRIENT_EFFECT.get(past_crop, {"N": 0, "P": 0, "K": 0, "pH": 0})
    return {
        "N_ppm": max(20, BASELINE_SOIL["N_ppm"] + effect["N"]),
        "P_ppm": max(5,  BASELINE_SOIL["P_ppm"] + effect["P"]),
        "K_ppm": max(50, BASELINE_SOIL["K_ppm"] + effect["K"]),
        "pH":    round(BASELINE_SOIL["pH"] + effect["pH"], 2),
        "OM_pct": BASELINE_SOIL["OM_pct"],
    }


def describe_nutrient_status(soil: Dict[str, float]) -> List[str]:
    """Human-readable list of likely deficiencies."""
    notes = []
    if soil["N_ppm"] < 80:
        notes.append("Nitrogen (N) likely low")
    if soil["P_ppm"] < 15:
        notes.append("Phosphorus (P) likely low")
    if soil["K_ppm"] < 150:
        notes.append("Potassium (K) likely low")
    if soil["pH"] < 6.5:
        notes.append("Soil slightly acidic")
    elif soil["pH"] > 8.0:
        notes.append("Soil alkaline")
    if not notes:
        notes.append("Nutrient status appears moderate (soil test still recommended)")
    return notes


class MittiMitraEngine:
    def __init__(self, model_dir: str = "model_assets"):
        model_dir = Path(model_dir)
        self.model = pickle.load(open(model_dir / "xgb_crop_model.pkl", "rb"))
        self.scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
        self.target_encoder = pickle.load(open(model_dir / "target_encoder.pkl", "rb"))
        self.label_encoders = pickle.load(open(model_dir / "label_encoders.pkl", "rb"))
        self.feature_order = pickle.load(open(model_dir / "feature_order.pkl", "rb"))
        logging.info("MittiMitraEngine loaded successfully.")

    def run_inference(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction entry point.

        Expected user_inputs keys:
          - past_crop (str)
          - latitude, longitude (float)   [preferred]
          - temperature_c, humidity_pct, rainfall_mm (float)
          - optional: district, village, field_size
        """
        past_crop = str(user_inputs.get("past_crop", "Wheat"))
        lat = float(user_inputs.get("latitude", 28.47))          # Greater Noida default
        lon = float(user_inputs.get("longitude", 77.50))
        temp = float(user_inputs.get("temperature_c", 28.0))
        humidity = float(user_inputs.get("humidity_pct", 60.0))
        rainfall = float(user_inputs.get("rainfall_mm", 5.0))

        # Infer current soil status from past crop
        soil = infer_soil_nutrients(past_crop)
        nutrient_notes = describe_nutrient_status(soil)

        # Build feature vector in the exact order used during training
        feature_map = {
            "past_crop": past_crop,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "rainfall_mm": rainfall,
            "inferred_N_ppm": soil["N_ppm"],
            "inferred_P_ppm": soil["P_ppm"],
            "inferred_K_ppm": soil["K_ppm"],
            "inferred_pH": soil["pH"],
            "inferred_OM_pct": soil["OM_pct"],
        }

        # Encode categorical
        le = self.label_encoders["past_crop"]
        if past_crop in le.classes_:
            feature_map["past_crop"] = le.transform([past_crop])[0]
        else:
            feature_map["past_crop"] = 0  # fallback

        X = np.array([[feature_map[f] for f in self.feature_order]])
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        classes = self.target_encoder.classes_
        ranked = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 2-4 recommendations
        top_k = min(4, len(ranked))
        recommendations = []
        for crop, conf in ranked[:top_k]:
            recommendations.append({
                "crop": crop,
                "confidence": round(float(conf) * 100, 1),
                "growing_requirements": GROWING_REQUIREMENTS.get(crop, "Follow local agronomic advice."),
            })

        result = {
            "recommendations": recommendations,
            "primary_crop": recommendations[0]["crop"],
            "primary_confidence": recommendations[0]["confidence"],
            "inferred_soil": soil,
            "nutrient_status": nutrient_notes,
            "inputs_used": {
                "past_crop": past_crop,
                "latitude": lat,
                "longitude": lon,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "rainfall_mm": rainfall,
            },
        }
        logging.info(f"Inference complete. Primary recommendation: {result['primary_crop']}")
        return result


# ---------------------------------------------------------------------------
# Quick local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing updated prediction engine...")
    engine = MittiMitraEngine()
    test_input = {
        "past_crop": "Wheat",
        "latitude": 28.4744,
        "longitude": 77.5040,
        "temperature_c": 26.5,
        "humidity_pct": 58,
        "rainfall_mm": 2.0,
    }
    out = engine.run_inference(test_input)
    print("\n=== RESULT ===")
    for i, rec in enumerate(out["recommendations"], 1):
        print(f"{i}. {rec['crop']} ({rec['confidence']}%)")
        print(f"   → {rec['growing_requirements']}")
    print("\nNutrient status:", out["nutrient_status"])
    print("Inferred soil:", out["inferred_soil"])
