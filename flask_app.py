"""
Mitti Mitra - Flask Backend (Phase 3)
Serves the modern farmer UI and connects to the updated prediction engine.
"""
import os
import json
from flask import Flask, render_template, request, jsonify
from prediction_engine import MittiMitraEngine

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load engine once
try:
    engine = MittiMitraEngine()
    ENGINE_READY = True
except Exception as e:
    print(f"Engine load warning: {e}")
    engine = None
    ENGINE_READY = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if not ENGINE_READY:
        return jsonify({"error": "Model not loaded. Run train_engine_v2.py first."}), 500

    try:
        result = engine.run_inference(data)
        # Add a simple microbiome score derived from soil + rotation
        soil = result.get("inferred_soil", {})
        n_score = min(100, max(30, soil.get("N_ppm", 80) * 0.6))
        diversity_penalty = 15 if data.get("past_crop") in ["Wheat", "Rice"] else 0
        microbiome_score = int(max(40, min(95, n_score - diversity_penalty + 20)))
        result["microbiome_score"] = microbiome_score
        result["microbiome_status"] = (
            "Theek-thaak — diversity kam hai" if microbiome_score < 75
            else "Accha — soil healthy"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
