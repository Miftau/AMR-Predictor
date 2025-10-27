# app.py (Enhanced)
import os
import uuid
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify
)
from model_utils import load_models, make_combined_text, vectorize_input, predict_all
from scipy.sparse import issparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
STATIC_IMG = "static/images"
ALLOWED_EXTENSIONS = {"csv", "tsv", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.secret_key = "supersecret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load pre-trained models
MODELS = load_models()


# --- Helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_confusion_plot(y_true, y_pred, title, save_path):
    """Generate and save confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# --- Routes ---

@app.route("/")
def index():
    """Homepage with upload and manual entry options."""
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    """Preview uploaded file and select columns for prediction."""
    if "file" not in request.files:
        flash("No file part provided.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Detect delimiter
        sep = "\t" if filename.endswith(".tsv") or filename.endswith(".txt") else ","
        df = pd.read_csv(file_path, sep=sep, nrows=20, engine="python")

        # Show first few rows and columns to user
        columns = df.columns.tolist()
        return render_template(
            "preview.html",
            filename=filename,
            columns=columns,
            preview=df.head().to_html(classes="table-auto w-full text-xs")
        )

    flash("Invalid file format.", "danger")
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    """Perform predictions on uploaded file after column mapping."""
    filename = request.form.get("filename")
    col_selection = request.form.getlist("selected_columns")

    if not filename or not col_selection:
        flash("Please select at least one column.", "danger")
        return redirect(url_for("index"))

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    sep = "\t" if filename.endswith(".tsv") or filename.endswith(".txt") else ","
    df = pd.read_csv(file_path, sep=sep, engine="python")

    # Combine selected columns
    df["combined_text"] = df[col_selection].astype(str).apply(lambda x: " ".join(x), axis=1)

    vectorizer = MODELS.get("vectorizer")
    X_vec = vectorizer.transform(df["combined_text"])
    preds = predict_all(MODELS, X_vec)

    # Append predictions
    for k, v in preds.items():
        df[k] = v

    output_path = os.path.join(UPLOAD_FOLDER, f"predicted_{filename}")
    df.to_csv(output_path, index=False)

    flash("Predictions generated successfully!", "success")
    return render_template("results.html", tables=[df.head(20).to_html(classes="table-auto text-xs w-full")],
            filename=os.path.basename(output_path))


@app.route("/dashboard")
def dashboard():
    """Show model performance metrics and visualizations."""
    metrics_path = "models/model_metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics_html = metrics_df.to_html(classes="table-auto text-sm w-full")
    else:
        metrics_html = "<p>No metrics available yet.</p>"

    # Display saved plots (if available)
    available_plots = [f for f in os.listdir(STATIC_IMG) if f.endswith(".png")]
    return render_template("dashboard.html",
                           metrics_html=metrics_html,
                           plots=available_plots)


@app.route('/explain/<int:sample_id>')
def explain(sample_id):
    # Example mock data to simulate explanation
    features = ['model_name', 'aro_name', 'amr_family', 'card_short_name']
    X_sample = np.random.randn(1, len(features))

    # Mock SHAP values
    shap_values = np.random.randn(1, len(features)) * 0.5
    base_value = np.random.randn()

    # Create a SHAP-like waterfall chart using Plotly
    fig = go.Figure()

    colors = ['#EF4444' if val < 0 else '#10B981' for val in shap_values[0]]
    cumulative = base_value
    positions = np.arange(len(features))

    fig.add_trace(go.Bar(
        x=features,
        y=shap_values[0],
        marker_color=colors,
        text=[f"{v:.3f}" for v in shap_values[0]],
        textposition='auto',
        name="SHAP Contribution"
    ))

    fig.update_layout(
        title=f"Feature Impact on AMR Prediction (Sample {sample_id})",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        template="plotly_white",
        showlegend=False,
        height=500
    )

    plot_html = fig.to_html(full_html=False)

    return render_template('explain.html', plot_html=plot_html, sample_id=sample_id)

@app.route("/downloads/<name>")
def download_file(name):
    return send_from_directory(UPLOAD_FOLDER, name, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
