# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
import uuid
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")  # backend for saving PNGs without display
import matplotlib.pyplot as plt
import shap
from model_utils import load_models, make_combined_text, vectorize_input, predict_all
from scipy.sparse import issparse

UPLOAD_FOLDER = "uploads"
STATIC_IMG = "static/images"
ALLOWED_EXTENSIONS = {"csv", "tsv", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)
os.makedirs("models", exist_ok=True)  # ensure folder exists

app = Flask(__name__)
app.secret_key = "change-me-to-a-secret"  # replace with env var in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load models once
MODELS = load_models()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Manual input form fields expected:
    # model_name, aro_name, amr_gene_family, card_short_name
    form = request.form
    record = {
        "Model Name": form.get("model_name", ""),
        "ARO Name": form.get("aro_name", ""),
        "AMR Gene Family": form.get("amr_family", ""),
        "CARD Short Name": form.get("card_short_name", "")
    }
    combined = make_combined_text(record)
    vectorizer = MODELS.get('vectorizer')
    if vectorizer is None:
        flash("Vectorizer not found on server.", "danger")
        return redirect(url_for("index"))

    x_vec = vectorize_input(vectorizer, combined)
    preds = predict_all(MODELS, x_vec)

    # Build results for single sample
    def first_or_none(val):
        if val is None:
            return None
        return val[0] if isinstance(val, list) or hasattr(val, "__len__") else val

    result = {
        "input": record,
        "rf_drug_pred": first_or_none(preds.get('rf_drug_pred')),
        "rf_drug_conf": first_or_none(preds.get('rf_drug_conf')),
        "rf_mech_pred": first_or_none(preds.get('rf_mech_pred')),
        "rf_mech_conf": first_or_none(preds.get('rf_mech_conf')),
        "deep_drug_pred": first_or_none(preds.get('deep_drug_pred')),
        "deep_drug_conf": first_or_none(preds.get('deep_drug_conf')),
        "deep_mech_pred": first_or_none(preds.get('deep_mech_pred')),
        "deep_mech_conf": first_or_none(preds.get('deep_mech_conf'))
    }

    # Save the single sample X to static for SHAP if needed
    uid = str(uuid.uuid4())[:8]
    sample_path = os.path.join(STATIC_IMG, f"sample_{uid}.pkl")
    pd.Series([combined]).to_pickle(sample_path)

    return render_template("results.html", result=result, sample_id=uid)

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("index"))
    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Read file (detect delimiter)
        if filename.lower().endswith(".tsv") or filename.lower().endswith(".txt"):
            df = pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python")
        else:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")

        # Ensure text features exist
        text_cols = [c for c in ["Model Name", "ARO Name", "AMR Gene Family", "CARD Short Name"] if c in df.columns]
        if not text_cols:
            flash("Uploaded file doesn't contain required text columns.", "danger")
            return redirect(url_for("index"))

        # Build combined_text column, vectorize and predict in batch
        df['combined_text'] = df[text_cols].astype(str).apply(lambda x: " ".join(x), axis=1)
        vectorizer = MODELS.get('vectorizer')

        X_vec = vectorizer.transform(df['combined_text'])
        preds = predict_all(MODELS, X_vec)

        # Assemble dataframe with predictions
        out = df.copy()
        if preds.get('rf_drug_pred') is not None:
            out['rf_drug_pred'] = preds['rf_drug_pred']
            out['rf_drug_conf'] = preds['rf_drug_conf']
        if preds.get('rf_mech_pred') is not None:
            out['rf_mech_pred'] = preds['rf_mech_pred']
            out['rf_mech_conf'] = preds['rf_mech_conf']
        if preds.get('deep_drug_pred') is not None:
            out['deep_drug_pred'] = preds['deep_drug_pred']
            out['deep_drug_conf'] = preds['deep_drug_conf']
            out['deep_mech_pred'] = preds['deep_mech_pred']
            out['deep_mech_conf'] = preds['deep_mech_conf']

        # Save predictions CSV
        out_filename = f"predictions_{filename.rsplit('.',1)[0]}.csv"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)
        out.to_csv(out_path, index=False)

        flash(f"Batch prediction complete — saved to {out_filename}", "success")
        return redirect(url_for('download_file', name=out_filename))
    else:
        flash("File type not allowed.", "danger")
        return redirect(url_for("index"))

@app.route('/downloads/<name>')
def download_file(name):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)

# SHAP explain endpoint for a single sample (from manual prediction or uploaded)
@app.route("/explain/<sample_id>", methods=["GET"])
def explain(sample_id):
    # load sample text from static/images/sample_<id>.pkl (created in /predict)
    sample_path = os.path.join(STATIC_IMG, f"sample_{sample_id}.pkl")
    if not os.path.exists(sample_path):
        flash("Sample not found for explanation.", "danger")
        return redirect(url_for("index"))

    import pandas as pd
    combined = pd.read_pickle(sample_path)[0]

    vectorizer = MODELS.get('vectorizer')
    rf_drug = MODELS.get('rf_drug')
    rf_mech = MODELS.get('rf_mech')

    if vectorizer is None or rf_drug is None:
        flash("Required models/encoders not available for explanation.", "danger")
        return redirect(url_for("index"))

    x_vec = vectorizer.transform([combined])

    # For Tree explainability (fast): use TreeExplainer on rf_drug (Drug Class)
    try:
        explainer = shap.TreeExplainer(rf_drug)
        X_for_shap = x_vec.toarray() if issparse(x_vec) else x_vec
        shap_values = explainer.shap_values(X_for_shap)
        # shap_values can be list (one per class) or ndarray; choose a class index (predicted)
        pred_idx = rf_drug.predict(x_vec)[0]
        # create waterfall/force plot for the predicted class
        img_name = f"shap_force_{sample_id}.png"
        img_path = os.path.join(STATIC_IMG, img_name)

        # Build single-sample explanation object
        # For new SHAP versions, use Explainer(X) pattern to obtain a ShapValues-like object
        # We'll create a waterfall plot using shap.plots.waterfall with shap.Explanation

        # compute shap.Explanation
        try:
            shap_exp = explainer(X_for_shap)
            # shap_exp.values shape -> (n_samples, n_features) or (n_samples, n_classes, n_features)
            # For multi-class tree explainer, shap_exp.values may be (n_samples, n_classes, n_features)
            # select predicted class index
            if shap_exp.values.ndim == 3:
                single_vals = shap_exp.values[0, pred_idx, :]
                single_base = shap_exp.base_values[0, pred_idx]
            else:
                single_vals = shap_exp.values[0, :]
                single_base = shap_exp.base_values[0]
            # Build a shap.Explanation to pass to waterfall plot
            explanation = shap.Explanation(values=single_vals,
                                           base_values=single_base,
                                           data=X_for_shap[0],
                                           feature_names=vectorizer.get_feature_names_out())
            plt.figure(figsize=(8,5))
            shap.plots.waterfall(explanation, show=False)
            plt.savefig(img_path, bbox_inches="tight", dpi=200)
            plt.close()
        except Exception:
            # fallback: create a bar summary plot for feature importances
            plt.figure(figsize=(10,4))
            try:
                # get mean absolute SHAP for each feature across classes
                if isinstance(shap_values, list):
                    # average across classes
                    shap_arr = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
                else:
                    shap_arr = np.abs(shap_values[0])
                feat_names = vectorizer.get_feature_names_out()
                # get top 20 features
                import numpy as np
                top_idx = np.argsort(shap_arr)[-20:][::-1]
                plt.barh(range(len(top_idx)), shap_arr[top_idx][::-1])
                plt.yticks(range(len(top_idx)), feat_names[top_idx][::-1])
                plt.title("Top SHAP features (approx.)")
                plt.tight_layout()
                plt.savefig(img_path, bbox_inches="tight", dpi=200)
                plt.close()
            except Exception as e:
                print("SHAP fallback plot failed:", e)
                flash("SHAP explanation generation failed.", "warning")
                return redirect(url_for("index"))

        return render_template("explain.html", sample_text=combined, image_file=img_name)
    except Exception as e:
        print("SHAP explanation error:", e)
        flash("SHAP explanation failed (see server logs).", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
