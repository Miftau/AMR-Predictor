# train_amr_model.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
from keras import layers, models, callbacks

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/amr_gene_data.csv"

# ========== STEP 1: LOAD DATA ==========
try:
    df = pd.read_csv(DATA_PATH, sep='\t', on_bad_lines='skip', engine='python')
    print(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# ========== STEP 2: CLEAN ==========
def clean_amr_dataset(df):
    cols_to_drop = [
        "ARO Accession", "CVTERM ID", "Model Sequence ID",
        "Model ID", "Protein Accession", "DNA Accession"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df = df.dropna(subset=["Drug Class", "Resistance Mechanism"])
    df = df.drop_duplicates()
    return df

df = clean_amr_dataset(df)
print("✅ Cleaned dataset shape:", df.shape)

# ========== STEP 3: VISUALIZATION ==========
plt.figure(figsize=(10, 5))
sns.countplot(y="Drug Class", data=df, order=df["Drug Class"].value_counts().index)
plt.title("Distribution of Drug Classes")
plt.tight_layout()
plt.savefig("drug_class_distribution.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(y="Resistance Mechanism", data=df, order=df["Resistance Mechanism"].value_counts().index)
plt.title("Distribution of Resistance Mechanisms")
plt.tight_layout()
plt.savefig("resistance_mechanism_distribution.png")
plt.close()

# ========== STEP 4: FEATURE PREPARATION ==========
text_features = [col for col in ["Model Name", "ARO Name", "AMR Gene Family", "CARD Short Name"] if col in df.columns]
df["combined_text"] = df[text_features].astype(str).apply(lambda x: " ".join(x), axis=1)

# Encode targets
le_drug = LabelEncoder()
le_mech = LabelEncoder()
df["drug_encoded"] = le_drug.fit_transform(df["Drug Class"])
df["mech_encoded"] = le_mech.fit_transform(df["Resistance Mechanism"])

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["combined_text"])
y1 = df["drug_encoded"].values
y2 = df["mech_encoded"].values

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, test_size=0.2, random_state=42
)

# ========== STEP 5: RANDOM FOREST ==========
print("\n🌲 Training RandomForest Models...")

rf_params = {
    'n_estimators': [100],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

rf_drug = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', verbose=1)
rf_mech = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', verbose=1)

rf_drug.fit(X_train, y1_train)
rf_mech.fit(X_train, y2_train)

print("✅ RF Drug Class Best Params:", rf_drug.best_params_)
print("✅ RF Mech Best Params:", rf_mech.best_params_)

y1_pred = rf_drug.predict(X_test)
y2_pred = rf_mech.predict(X_test)

print("\n📊 RF Drug Class Report:")
print(classification_report(y1_test, y1_pred))
print("Accuracy:", accuracy_score(y1_test, y1_pred))

print("\n📊 RF Resistance Mechanism Report:")
print(classification_report(y2_test, y2_pred))
print("Accuracy:", accuracy_score(y2_test, y2_pred))

# ========== STEP 6: DEEP LEARNING MODEL ==========
print("\n🧠 Training Deep Learning Model...")

input_dim = X_train.shape[1]
n_drug_classes = len(np.unique(y1))
n_mech_classes = len(np.unique(y2))

# Convert to dense for TensorFlow
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Define model
input_layer = layers.Input(shape=(input_dim,))
x = layers.Dense(512, activation='relu')(input_layer)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Two output heads
drug_output = layers.Dense(n_drug_classes, activation='softmax', name="drug_output")(x)
mech_output = layers.Dense(n_mech_classes, activation='softmax', name="mech_output")(x)

model = models.Model(inputs=input_layer, outputs=[drug_output, mech_output])
model.compile(
    optimizer='adam',
    loss={
        'drug_output': 'sparse_categorical_crossentropy',
        'mech_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'drug_output': ['accuracy'],
        'mech_output': ['accuracy']
    }
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_dense,
    {'drug_output': y1_train, 'mech_output': y2_train},
    validation_data=(X_test_dense, {'drug_output': y1_test, 'mech_output': y2_test}),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate
scores = model.evaluate(X_test_dense, {'drug_output': y1_test, 'mech_output': y2_test})
print(f"🧪 Deep Learning Test Loss: {scores[0]:.4f}")
print(f"🧪 Deep Learning Drug Class Accuracy: {scores[3]:.4f}")
print(f"🧪 Deep Learning Mechanism Accuracy: {scores[4]:.4f}")

# Plot training curves
plt.figure(figsize=(10,4))
plt.plot(history.history['drug_output_accuracy'], label='Train Drug Acc')
plt.plot(history.history['val_drug_output_accuracy'], label='Val Drug Acc')
plt.plot(history.history['mech_output_accuracy'], label='Train Mech Acc')
plt.plot(history.history['val_mech_output_accuracy'], label='Val Mech Acc')
plt.legend()
plt.title("Training & Validation Accuracy")
plt.tight_layout()
plt.savefig("deep_learning_training_curve.png")
plt.close()

# ========== STEP 7: SAVE ALL MODELS ==========
joblib.dump(rf_drug.best_estimator_, "models/rf_drug_model.pkl")
joblib.dump(rf_mech.best_estimator_, "models/rf_mech_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(le_drug, "models/label_encoder_drug.pkl")
joblib.dump(le_mech, "models/label_encoder_mech.pkl")
model.save("models/amr_deep_model.h5")

print("\n✅ All models saved successfully in /models/")
