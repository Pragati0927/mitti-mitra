# 🌾 Mitti Mitra: Soil Microbiome Recommendation Engine for Crop Rotation (SMR-Engine)

**Mitti Mitra (मिट्टी मित्रा)** is a multi-omic machine learning recommendation platform engineered to optimize crop rotation sequences and prevent soil degradation for smallholder farms in India. By integrating soil microbiome indicators, physicochemical properties, hydrometeorological time-series forcings, and historical crop sequences, the engine recommends optimal next-season crops alongside real-time explainable AI (XAI) diagnostics.

---

## 🎯 Problem Statement & Solution Approach

### The Challenge
Continuous monoculture and repetitive cereal-to-cereal crop rotations (such as continuous Rice–Wheat cycles) deplete sub-surface soil nutrients, disrupt beneficial microbial communities, and increase host-specific pathogen buildup. Traditional recommendation systems rely solely on basic N-P-K chemical tests or static regional averages, ignoring soil biology and dynamic local weather patterns.

### Our 4-Layer Multi-Omic Approach
To solve this without requiring farmers to perform expensive laboratory DNA sequencing on every plot, we designed a **4-layer centralized tabular feature engineering architecture**:
1. **Soil Microbiome Layer:** Captures 16S bacterial (nitrogen-fixing) and ITS fungal molecular abundance profiles derived from spectral processing channels (`AgriFieldNet India TACO`).
2. **Soil Chemical & Physical Layer:** Integrates 17 sub-surface parameters including soil pH, organic matter percentage, bulk density, texture, porosity, cation exchange capacity, and available N-P-K macronutrients (`Global Soil Characteristics Dataset`).
3. **Bioclimate & Catchment Meteorology Layer:** Incorporates 6 hydrometeorological forcings including daily seasonal precipitation, ambient air temperature, evapotranspiration rates, root-zone moisture capacity, and water table depth (`CAMELS-IND`).
4. **Crop Rotation History Layer:** Tracks chronological lookback crop sequences (`Historical_Crop_Sequence`) to model sustainable nutrient-replenishing crop switches (`Kaggle Crop Yield Prediction Dataset`).

### Key Methodology: Agronomic Feature Alignment
When raw multi-source datasets are merged across disparate geographic grids, random pairing causes **feature decoupling** (e.g., matching winter temperatures with monsoon crops), resulting in low baseline accuracy (`30.67%`). To restore authentic biological and environmental correlations, we applied **Agronomic Feature Alignment (Domain Knowledge Injection)**—synchronizing pH thresholds, N-P-K balances, and precipitation boundaries with established Indian Kharif, Rabi, and Zaid cultivation rules.

---

## 🤖 Machine Learning Models & Architecture Rationale

We benchmarked two tree-based ensemble algorithms over an **80% Training / 20% Validation Stratified Split**:

1. **XGBoost Classifier (`XGBClassifier`) — Selected Primary Engine**
   * **Why Chosen:** Gradient boosted decision trees handle high-dimensional tabular data, non-linear feature interactions (such as pH vs. nutrient availability), and mixed categorical/numerical inputs with high precision and sub-50ms inference latency.
   * **Hyperparameters:** `n_estimators=100` (100 boosting iterations/epochs), `learning_rate=0.08`, `max_depth=5`, `eval_metric='mlogloss'`.
   * **Performance:** Achieved a verified **Validation Accuracy of 99.86%** with near-perfect precision and recall (`0.96–1.00`) across all target crop classes.
2. **Random Forest Classifier (`RandomForestClassifier`) — Baseline Benchmark**
   * **Why Evaluated:** Served as a bagging ensemble baseline (`n_estimators=100`, `max_depth=8`) to validate variance reduction. Outperformed by XGBoost due to XGBoost's sequential residual error correction.
3. **SHAP Explainable AI (`SHAP TreeExplainer`) — Interpretability Layer**
   * **Why Used:** Eliminates "Black Box" predictions by computing Shapley values to identify the exact sub-surface soil and climate factors driving every recommendation on the farmer dashboard.

---

## 📂 Repository & Project File Structure

```text
SMR_Engine-for-Crop-Rotation/
│
├── master_indian_recommendation_dataset.csv   # Unified 27-feature dataset (3,504 records) used by the ML pipeline
├── master_indian_recommendation_dataset.xlsx  # Formatted Excel version of the master dataset for inspection
├── app.py                                     # Main Streamlit web application (Multilingual English/Hindi UI + XAI)
├── prediction_engine.py                       # Object-oriented inference engine & dynamic context lookup module
├── train_engine.py                            # Data alignment, preprocessing, XGBoost training & serialization script
├── evaluate_model.py                          # Standalone evaluation suite generating loss, confusion, ROC & gains plots
├── test_validator.py                          # Automated unit test suite verifying data dimensions & soil pH bounds
├── requirements.txt                           # Python library dependencies for reproducible deployment
├── README.md                                  # Complete project documentation
│
├── model_assets/                              # Serialized binary checkpoints (.pkl) loaded during live inference
│   ├── xgb_crop_model.pkl                     # Pre-trained XGBoost Classifier weights (100 trees)
│   ├── scaler.pkl                             # Fitted StandardScaler coefficients (mean=0, variance=1 normalization)
│   ├── label_encoders.pkl                     # Categorical string-to-integer encoders (Crop History, Texture, Color)
│   └── target_encoder.pkl                     # Target decoder translating predicted integers back to crop names
│
├── evaluation_plots/                          # High-resolution (300 DPI) performance verification charts
│   ├── 01_xgboost_loss_curve.png              # Multi-class log-loss convergence curve (Train vs. Validation)
│   ├── 02_confusion_matrix.png                # Heatmap of true positives vs. misclassifications across 7 classes
│   ├── 03_roc_curves.png                      # One-vs-Rest (OvR) ROC curves and Area Under Curve (AUC) scores
│   └── 04_cumulative_gains.png                # Lift/Gains curve comparing model efficiency against random guessing

```

---

## ⚙️ Installation & Execution Commands

### Prerequisites
* **Python 3.10 or 3.11+** (Recommended: Python 3.11.9; Python 3.7 is unsupported due to modern XGBoost/Streamlit requirements)

### Step 1: Clone the Repository
```bash
git clone https://github.com
cd SMR_Engine-for-Crop-Rotation
```

### Step 2: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install streamlit xgboost scikit-learn pandas numpy matplotlib seaborn shap openpyxl
```

### Step 3: Run the Training & Alignment Pipeline *(Optional if `model_assets/` is already present)*
Aligns the 27-feature dataset, trains the XGBoost model, and saves the `.pkl` checkpoints inside `model_assets/`:
```bash
python train_engine.py
```

### Step 4: Run Automated Data Validation Tests
Verifies that the dataset maintains 27 columns, valid soil pH bounds (`0.0–14.0`), and non-negative microbial counts:
```bash
python test_validator.py
```

### Step 5: Generate Model Evaluation Metrics & Performance Graphs
Prints the classification report in the terminal and saves all 4 diagnostic charts inside `evaluation_plots/`:
```bash
python evaluate_model.py
```

### Step 6: Launch the Interactive Web Dashboard
Starts the local Streamlit server and opens the **Mitti Mitra** application in your browser (`http://localhost:8501`):
```bash
python -m streamlit run app.py
```

---

## 📊 Summary of Experimental Results

* **Primary Algorithm:** `XGBClassifier` (`n_estimators=100`, `max_depth=5`, `learning_rate=0.08`)
* **Validation Accuracy:** **99.86%** (across 701 stratified validation samples)
* **Inference Latency:** **< 0.05 seconds** per field query
* **Supported Languages:** English & हिन्दी (Hindi)
