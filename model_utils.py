# model_utils.py
import os
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.sparse import issparse

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# expected filenames (adjust if different)
RF_DRUG_PATH = os.path.join(MODEL_DIR, "rf_drug_model.pkl")
RF_MECH_PATH = os.path.join(MODEL_DIR, "rf_mech_model.pkl")
VECTOR_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LE_DRUG_PATH = os.path.join(MODEL_DIR, "label_encoder_drug.pkl")
LE_MECH_PATH = os.path.join(MODEL_DIR, "label_encoder_mech.pkl")
DEEP_MODEL_PATH = os.path.join(MODEL_DIR, "amr_deep_model.h5")  # optional

# Load models
def load_models():
    models = {
        'rf_drug': (
            joblib.load(RF_DRUG_PATH) if os.path.exists(RF_DRUG_PATH) else None
        )
    }
    if os.path.exists(RF_MECH_PATH):
        models['rf_mech'] = joblib.load(RF_MECH_PATH)
    else:
        models['rf_mech'] = None

    if os.path.exists(VECTOR_PATH):
        models['vectorizer'] = joblib.load(VECTOR_PATH)
    else:
        models['vectorizer'] = None

    if os.path.exists(LE_DRUG_PATH):
        models['le_drug'] = joblib.load(LE_DRUG_PATH)
    else:
        models['le_drug'] = None

    if os.path.exists(LE_MECH_PATH):
        models['le_mech'] = joblib.load(LE_MECH_PATH)
    else:
        models['le_mech'] = None

    if os.path.exists(DEEP_MODEL_PATH):
        try:
            models['deep'] = load_model(DEEP_MODEL_PATH)
        except Exception as e:
            print("Warning: could not load deep model:", e)
            models['deep'] = None
    else:
        models['deep'] = None

    return models

# Combine input fields into the same combined_text as training used
def make_combined_text(record):
    # expects dict-like with possible keys: Model Name, ARO Name, AMR Gene Family, CARD Short Name
    fields = []
    fields.extend(
        str(record[k])
        for k in [
            "Model Name",
            "ARO Name",
            "AMR Gene Family",
            "CARD Short Name",
        ]
        if k in record and record[k] is not None
    )
    return " ".join(fields)

# Given a single input dict, return vectorized X (suitable for RF and deep)
def vectorize_input(vectorizer, input_text):
    return vectorizer.transform([input_text])

# Predict function: accepts vectorized X (sparse or dense)
def predict_all(models, x_vec):
    results = {}
    # RF predictions
    rf_drug = models.get('rf_drug')
    rf_mech = models.get('rf_mech')
    le_drug = models.get('le_drug')
    le_mech = models.get('le_mech')
    deep = models.get('deep')

    # Ensure dense for deep model
    x_for_deep = x_vec.toarray() if issparse(x_vec) else x_vec

    if rf_drug:
        preds = rf_drug.predict(x_vec)
        probs = rf_drug.predict_proba(x_vec)
        results['rf_drug_pred'] = le_drug.inverse_transform(preds) if le_drug else preds
        results['rf_drug_conf'] = np.max(probs, axis=1).tolist()
    else:
        results['rf_drug_pred'] = None
        results['rf_drug_conf'] = None

    if rf_mech:
        preds = rf_mech.predict(x_vec)
        probs = rf_mech.predict_proba(x_vec)
        results['rf_mech_pred'] = le_mech.inverse_transform(preds) if le_mech else preds
        results['rf_mech_conf'] = np.max(probs, axis=1).tolist()
    else:
        results['rf_mech_pred'] = None
        results['rf_mech_conf'] = None

    if deep:
        # Deep model is multi-output softmax heads
        preds = deep.predict(x_for_deep)
        # preds is list [drug_probs, mech_probs] or array depending on Keras version
        if isinstance(preds, list) and len(preds) >= 2:
            drug_probs = preds[0]
            mech_probs = preds[1]
            drug_idx = np.argmax(drug_probs, axis=1)
            mech_idx = np.argmax(mech_probs, axis=1)
            results['deep_drug_pred'] = models['le_drug'].inverse_transform(drug_idx) if models.get('le_drug') else drug_idx
            results['deep_drug_conf'] = np.max(drug_probs, axis=1).tolist()
            results['deep_mech_pred'] = models['le_mech'].inverse_transform(mech_idx) if models.get('le_mech') else mech_idx
            results['deep_mech_conf'] = np.max(mech_probs, axis=1).tolist()
        else:
            # fallback if predict returns numpy array with two heads concatenated (rare)
            results['deep_drug_pred'] = None
            results['deep_mech_pred'] = None
    else:
        results['deep_drug_pred'] = None
        results['deep_drug_conf'] = None
        results['deep_mech_pred'] = None
        results['deep_mech_conf'] = None

    return results
