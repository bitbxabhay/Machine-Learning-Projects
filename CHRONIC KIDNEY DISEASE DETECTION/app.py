from flask import Flask, abort, jsonify, render_template, request, send_from_directory
import joblib
import numpy as np
import os
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=PROJECT_DIR, static_folder=None)

model = joblib.load(os.path.join(PROJECT_DIR, "lr_model_ckd.pkl"))
scaler = joblib.load(os.path.join(PROJECT_DIR, "scaler.pkl"))
MODEL_FEATURES = joblib.load(os.path.join(PROJECT_DIR, "columns.pkl"))
CKD_CLASS = 0
NUMERIC_FEATURES = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
                    "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORY_FEATURES = {
    "rbc": ("rbc_normal", {"abnormal": 0, "normal": 1}),
    "pc": ("pc_normal", {"abnormal": 0, "normal": 1}),
    "pcc": ("pcc_present", {"notpresent": 0, "present": 1}),
    "ba": ("ba_present", {"notpresent": 0, "present": 1}),
    "htn": ("htn_yes", {"no": 0, "yes": 1}),
    "dm": ("dm_yes", {"no": 0, "yes": 1}),
    "cad": ("cad_yes", {"no": 0, "yes": 1}),
    "appet": ("appet_poor", {"good": 0, "poor": 1}),
    "pe": ("pe_yes", {"no": 0, "yes": 1}),
    "ane": ("ane_yes", {"no": 0, "yes": 1}),
}

if list(scaler.feature_names_in_) != MODEL_FEATURES:
    raise ValueError("Scaler feature order does not match columns.pkl")
if model.n_features_in_ != len(MODEL_FEATURES):
    raise ValueError("Model feature count does not match columns.pkl")
if CKD_CLASS not in model.classes_:
    raise ValueError(f"Model does not contain the CKD class {CKD_CLASS}")


@app.route("/kidney.png")
def kidney_image():
    return send_from_directory(PROJECT_DIR, "kidney.png")


@app.route("/kidney-neon.svg")
def kidney_neon_image():
    return send_from_directory(PROJECT_DIR, "kidney-neon.svg")


@app.route("/logo.png")
def logo_image():
    return send_from_directory(PROJECT_DIR, "logo.png")


@app.route("/background.png")
def background_image():
    return send_from_directory(PROJECT_DIR, "background.png")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request must contain a JSON object."}), 400

    required_features = NUMERIC_FEATURES + list(CATEGORY_FEATURES)
    missing_features = [feature for feature in required_features if feature not in data]
    if missing_features:
        return jsonify({"error": "Missing fields: " + ", ".join(missing_features)}), 400

    try:
        encoded = {feature: float(data[feature]) for feature in NUMERIC_FEATURES}
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid numeric input: {error}"}), 400

    if not np.isfinite(list(encoded.values())).all():
        return jsonify({"error": "Numeric inputs must be finite values."}), 400

    for feature, (model_feature, choices) in CATEGORY_FEATURES.items():
        value = str(data[feature]).strip().lower()
        if value not in choices:
            return jsonify({"error": f"Invalid value for {feature}: {data[feature]}"}), 400
        encoded[model_feature] = choices[value]

    x = pd.DataFrame([[encoded[feature] for feature in MODEL_FEATURES]],
                     columns=MODEL_FEATURES)
    x_scaled = scaler.transform(x)
    prediction = int(model.predict(x_scaled)[0])
    ckd_probability = None
    if hasattr(model, "predict_proba"):
        ckd_index = list(model.classes_).index(CKD_CLASS)
        ckd_probability = float(model.predict_proba(x_scaled)[0][ckd_index])

    return jsonify({
        "ckd": prediction == CKD_CLASS,
        "probability": ckd_probability,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)