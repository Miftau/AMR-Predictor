# explain_model_with_shap.py
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from keras.models import load_model
from scipy.sparse import issparse

# ----------------------------
# 1. Load trained model and encoders
# ----------------------------
print("🔹 Loading trained model and encoders...")
model = load_model("models/amr_deep_model.h5")

X_vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder_drug = joblib.load("models/label_encoder_drug.pkl")
label_encoder_mech = joblib.load("models/label_encoder_mech.pkl")

# ----------------------------
# 2. Load and preprocess sample data
# ----------------------------
print("🔹 Loading dataset sample for SHAP analysis...")
data = pd.read_csv("data/amr_gene_data.csv", sep="\t", on_bad_lines='skip')

# Drop columns not used during training
drop_cols = [
    "ARO Accession", "CVTERM ID", "Model Sequence ID", "Model ID",
    "Model Name", "ARO Name", "Protein Accession", "DNA Accession"
]
data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")

# Extract text or sequence feature used for prediction
# (Example: using "CARD Short Name")
if "CARD Short Name" not in data.columns:
    raise ValueError("❌ 'CARD Short Name' column missing in dataset!")
X_text = data["CARD Short Name"].astype(str)
X_transformed = X_vectorizer.transform(X_text)

# Select a small sample for SHAP explanation
sample_size = min(100, X_transformed.shape[0])
X_sample = X_transformed[:sample_size]

# ----------------------------
# 3. Initialize SHAP explainer
# ----------------------------
print("🔹 Initializing SHAP explainer (this may take a while)...")

# Define prediction function for Drug Class (output 0)
def predict_drug(x):
    preds = model.predict(x)
    return preds[0]  # first output

# Define prediction function for Resistance Mechanism (output 1)
def predict_mech(x):
    preds = model.predict(x)
    return preds[1]  # second output

# KernelExplainer background: take a small dense subset
if issparse(X_sample):
    background = X_sample[:10].toarray()
else:
    background = X_sample[:10]

explainer_drug = shap.KernelExplainer(predict_drug, background)
explainer_mech = shap.KernelExplainer(predict_mech, background)

# ----------------------------
# 4. Compute SHAP values
# ----------------------------
print("🔹 Computing SHAP values (this may take several minutes)...")
X_dense_sample = X_sample[:20].toarray() if issparse(X_sample) else X_sample[:20]

shap_values_drug = explainer_drug.shap_values(X_dense_sample, nsamples=50)
shap_values_mech = explainer_mech.shap_values(X_dense_sample, nsamples=50)

# ----------------------------
# 5. Visualize SHAP summary plots
# ----------------------------
print("🔹 Generating SHAP summary plots...")
feature_names = np.array(X_vectorizer.get_feature_names_out())

# Drug Class
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_drug, X_dense_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_drug_class.png", dpi=300)
plt.close()

# Resistance Mechanism
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_mech, X_dense_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_resistance_mechanism.png", dpi=300)
plt.close()

print("✅ Summary plots saved for both Drug Class and Resistance Mechanism.")

# ----------------------------
# 6. Force plots for both outputs (static)
# ----------------------------
print("🔹 Generating static force plots...")

try:
    shap.initjs()
except Exception:
    print("⚠️ Skipping JS initialization (IPython not available).")

# Drug Class
shap.force_plot(
    explainer_drug.expected_value,
    shap_values_drug,
    matplotlib=True,
    show=False
)
plt.title("Force Plot – Drug Class")
plt.savefig("shap_force_drug_class.png", dpi=300)
plt.close()

# Resistance Mechanism
shap.force_plot(
    explainer_mech.expected_value,
    shap_values_mech,
    matplotlib=True,
    show=False
)
plt.title("Force Plot – Resistance Mechanism")
plt.savefig("shap_force_resistance_mechanism.png", dpi=300)
plt.close()

print("🎉 SHAP force plots saved for both outputs.")
print("✅ SHAP analysis complete!")
