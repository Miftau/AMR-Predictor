# app.py (Enhanced)
import datetime
import os
import uuid
import joblib
import shap
import shap.maskers
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
from flask_mail import Mail, Message       
from model_utils import load_models, make_combined_text, vectorize_input, predict_all
from scipy.sparse import issparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from dotenv import load_dotenv  

load_dotenv()  # Load environment variables from .env file

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

# === Flask-Mail Configuration ===
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = ("AMR Predictor Contact", os.getenv("MAIL_USERNAME"))

mail = Mail(app)

@app.context_processor
def inject_now():
    """Add current year to all templates."""
    return {"current_year": datetime.datetime.now().year}

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
def home():
    """Homepage"""
    return render_template("home.html")

@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        if not name or not email or not message:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("contact"))

        try:
            msg = Message(
                subject=f"New Contact Form Message from {name}",
                recipients=[os.getenv("MAIL_DEFAULT_RECEIVER")],
                body=f"""
                You have received a new message from your website contact form.

                Name: {name}
                Email: {email}

                Message:
                {message}
                """
            )
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Error sending message: {str(e)}", "danger")

        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route("/form")
def form():
    """Upload or prediction form"""
    return render_template("form.html")


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
    """Perform predictions on uploaded file or manual input, and provide SHAP explanations."""
    filename = request.form.get("filename")
    col_selection = request.form.getlist("selected_columns")
    manual_text = request.form.get("manual_text")

    # Handle file-based batch predictions
    if filename:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        sep = "\t" if filename.endswith(".tsv") or filename.endswith(".txt") else ","
        df = pd.read_csv(file_path, sep=sep, engine="python")

        if not col_selection:
            flash("Please select at least one column.", "danger")
            return redirect(url_for("index"))

        # Combine selected columns
        df["combined_text"] = df[col_selection].astype(str).apply(lambda x: " ".join(x), axis=1)
        text_data = df["combined_text"]

    # Handle manual input
    elif manual_text:
        text_data = [manual_text]
        df = pd.DataFrame({"combined_text": text_data})

    else:
        flash("Please provide input data (either upload file or manual text).", "danger")
        return redirect(url_for("index"))

    # Load vectorizer and models
    vectorizer = MODELS.get("vectorizer")
    X_vec = vectorizer.transform(text_data)
    preds = predict_all(MODELS, X_vec)

    # Attach predictions to DataFrame
    for k, v in preds.items():
        df[k] = v

    # Generate SHAP explanations
    try:
        # Pick best available model for explanation
        model = MODELS.get("rf_drug") or MODELS.get("deep")
        vectorizer = MODELS.get("vectorizer")

        if model is None or vectorizer is None:
            print("⚠️ No model or vectorizer available for SHAP explanations.")
            shap_summary_path = None
        else:
            # Prepare input data
            X_sample = X_vec[:50]
            if issparse(X_sample):
                X_sample = X_sample.toarray()

            # Choose correct SHAP explainer
            if "forest" in str(type(model)).lower() or "tree" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model.predict_proba, X_sample)
            else:
                explainer = shap.Explainer(model.predict, X_sample)

            shap_values = explainer(X_sample)

            # Generate a summary plot
            shap_dir = os.path.join("static", "shap")
            os.makedirs(shap_dir, exist_ok=True)
            shap_summary_filename = f"shap_summary_{uuid.uuid4().hex}.png"
            shap_summary_path = os.path.join(shap_dir, shap_summary_filename)

            plt.figure(figsize=(8, 5))
            shap.summary_plot(shap_values, feature_names=getattr(vectorizer, "get_feature_names_out", lambda: None)(), show=False)
            plt.tight_layout()
            plt.savefig(shap_summary_path, bbox_inches="tight", dpi=200)
            plt.close()
    except Exception as e:
        print("❌ SHAP generation failed:", e)
        shap_summary_path = None

    # Save batch prediction results
    output_path = os.path.join(UPLOAD_FOLDER, f"predicted_{filename or 'manual_input.csv'}")
    df.to_csv(output_path, index=False)

    flash("Predictions and explanations generated successfully!", "success")
    return render_template(
        "results.html",
        tables=[df.head(20).to_html(classes="table-auto text-xs w-full")],
        filename=os.path.basename(output_path),
        shap_summary=os.path.basename(shap_summary_path)
    )


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

@app.route("/explain_batch/<batch_id>/<int:row_index>", methods=["GET"])
def explain_batch(batch_id, row_index):
    import pandas as pd
    batch_path = os.path.join(STATIC_IMG, f"batch_{batch_id}.pkl")
    if not os.path.exists(batch_path):
        flash("Batch file not found for explanation.", "danger")
        return redirect(url_for("index"))
    
    df = pd.read_pickle(batch_path)
    if row_index >= len(df):
        flash("Invalid row index for SHAP explanation.", "danger")
        return redirect(url_for("index"))
    
    # Get the text used for prediction
    combined = df.iloc[row_index]['combined_text']
    vectorizer = MODELS.get('vectorizer')
    rf_drug = MODELS.get('rf_drug')

    if vectorizer is None or rf_drug is None:
        flash("Model or vectorizer missing for explanation.", "danger")
        return redirect(url_for("index"))

    x_vec = vectorizer.transform([combined])

    try:
        explainer = shap.TreeExplainer(rf_drug)
        X_for_shap = x_vec.toarray() if issparse(x_vec) else x_vec
        shap_exp = explainer(X_for_shap)

        # Determine predicted class index
        pred_idx = rf_drug.predict(x_vec)[0]
        if shap_exp.values.ndim == 3:
            single_vals = shap_exp.values[0, pred_idx, :]
            single_base = shap_exp.base_values[0, pred_idx]
        else:
            single_vals = shap_exp.values[0, :]
            single_base = shap_exp.base_values[0]

        explanation = shap.Explanation(
            values=single_vals,
            base_values=single_base,
            data=X_for_shap[0],
            feature_names=vectorizer.get_feature_names_out()
        )

        img_name = f"shap_batch_{batch_id}_{row_index}.png"
        img_path = os.path.join(STATIC_IMG, img_name)
        plt.figure(figsize=(8,5))
        shap.plots.waterfall(explanation, show=False)
        plt.savefig(img_path, bbox_inches="tight", dpi=200)
        plt.close()

        return render_template("explain.html", sample_text=combined, image_file=img_name)

    except Exception as e:
        print("SHAP batch explanation error:", e)
        flash("Failed to generate SHAP explanation for this row.", "danger")
        return redirect(url_for("index"))


@app.route("/downloads/<name>")
def download_file(name):
    return send_from_directory(UPLOAD_FOLDER, name, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
