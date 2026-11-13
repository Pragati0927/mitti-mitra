import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

print("Loading dataset and saved model assets...")

# 1. Load your 27-column dataset and saved pipeline tools
df = pd.read_csv('master_indian_recommendation_dataset.csv')

with open('model_assets/xgb_crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_assets/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model_assets/target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)
with open('model_assets/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# 2. Prepare features (X) and target labels (Y)
target_col = 'Target_Recommended_Crop'
categorical_cols = ['Historical_Crop_Sequence', 'Texture', 'Soil_Color']

Y_raw = df[target_col].astype(str)
X_raw = df.drop(columns=[target_col, 'Soil_ID'], errors='ignore')

for col in categorical_cols:
    if col in X_raw.columns:
        le = label_encoders[col]
        X_raw[col] = X_raw[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

Y = target_encoder.transform(Y_raw)
class_names = target_encoder.classes_
n_classes = len(class_names)

X_scaled = scaler.transform(X_raw)

# 3. Split data (80% Train, 20% Test) matching your training setup
X_train, X_val, Y_train, Y_val = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
)

# 4. Run predictions and print accuracy report
Y_pred = model.predict(X_val)
Y_prob = model.predict_proba(X_val)
acc = accuracy_score(Y_val, Y_pred)

print(f"ACTIVE MODEL     : {type(model).__name__}")
print(f"EXACT ACCURACY   : {acc * 100:.2f}%")
print("\nCLASSIFICATION REPORT (Precision, Recall, F1-Score):")
print(classification_report(Y_val, Y_pred, target_names=class_names, zero_division=0))

# Create output folder for images
os.makedirs('evaluation_plots', exist_ok=True)
print("Generating 4 evaluation graphs...")

# ==============================================================================
# GRAPH 1: MULTI-CLASS LOG-LOSS CURVE
# ==============================================================================
# Re-fit a tracker clone on the exact split to record epoch-by-epoch loss history
from xgboost import XGBClassifier
tracker_model = XGBClassifier(
    n_estimators=100, learning_rate=0.08, max_depth=5, 
    random_state=42, eval_metric='mlogloss'
)
tracker_model.fit(
    X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_val, Y_val)],
    verbose=False
)
results = tracker_model.evals_result()
epochs = range(len(results['validation_0']['mlogloss']))

plt.figure(figsize=(8, 5))
plt.plot(epochs, results['validation_0']['mlogloss'], label='Training Loss', linewidth=2)
plt.plot(epochs, results['validation_1']['mlogloss'], label='Validation Loss', linewidth=2)
plt.title('XGBoost Multi-Class Log-Loss Convergence')
plt.xlabel('Boosting Iterations (Trees / Epochs)')
plt.ylabel('Multi-Class Log Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('evaluation_plots/01_xgboost_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# GRAPH 2: CONFUSION MATRIX HEATMAP
# ==============================================================================
cm = confusion_matrix(Y_val, Y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('XGBoost Multi-Class Confusion Matrix')
plt.xlabel('Predicted Crop')
plt.ylabel('Actual Ground Truth Crop')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('evaluation_plots/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# GRAPH 3: ONE-VS-REST (OvR) ROC CURVES
# ==============================================================================
Y_val_bin = label_binarize(Y_val, classes=np.arange(n_classes))

plt.figure(figsize=(9, 7))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(Y_val_bin[:, i], Y_prob[:, i]) # type: ignore
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)')
plt.xlim([0.0, 1.0]) # type: ignore
plt.ylim([0.0, 1.05]) # type: ignore
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('One-vs-Rest (OvR) ROC Curves per Crop')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('evaluation_plots/03_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# GRAPH 4: CUMULATIVE GAINS CHART
# ==============================================================================
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing Baseline')

for i in range(n_classes):
    targets = (Y_val == i).astype(int)
    if targets.sum() == 0:
        continue
    probs = Y_prob[:, i]
    df_gains = pd.DataFrame({'target': targets, 'prob': probs}).sort_values(by='prob', ascending=False)
    cum_gains = df_gains['target'].cumsum() / df_gains['target'].sum()
    pct_samples = np.arange(1, len(cum_gains) + 1) / len(cum_gains)
    plt.plot(pct_samples, cum_gains, linewidth=2, label=f'{class_names[i]}')

plt.xlabel('Percentage of Fields Evaluated')
plt.ylabel('Percentage of Correct Crops Found')
plt.title('Cumulative Gains Chart (Model Efficiency)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('evaluation_plots/04_cumulative_gains.png', dpi=300, bbox_inches='tight')
plt.close()

print("Done! All 4 graphs are saved inside the 'evaluation_plots/' folder.")
