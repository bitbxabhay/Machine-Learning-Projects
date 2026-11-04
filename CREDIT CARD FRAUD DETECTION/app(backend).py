"""
Credit Card 💳Fraud Detection - Flask backend
Run:  pip install flask flask-cors joblib numpy scikit-learn
      python app.py
Open: http://127.0.0.1:5000
"""
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------- CONFIG (apne hisaab se badlo) ----------------
MODEL_PATH = "KNN_creditcard.pkl"      # tumhara trained model (joblib / pickle)
SCALER_PATH = "scaler.pkl"            # agar Time/Amount scale kiya tha to e.g. "scaler.pkl"
THRESHOLD = 0.40              # classification threshold
# ----------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

model = joblib.load(os.path.join(BASE_DIR, MODEL_PATH))
scaler = joblib.load(os.path.join(BASE_DIR, SCALER_PATH)) if SCALER_PATH else None


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        time_ = float(data["Time"])
        amount = float(data["Amount"])
        v = [float(x) for x in data["features"]]
        if len(v) != 28:
            return jsonify(error="28 PCA features (V1-V28) chahiye."), 400

        # Kaggle creditcard.csv ka column order: Time, V1..V28, Amount
        row = np.array([[time_] + v + [amount]])

        # Agar tumne training me scaling ki thi to yahan same karo
        if scaler is not None:
            row = scaler.transform(row)

        prob = float(model.predict_proba(row)[0][1])
        return jsonify(
            fraud_probability=prob,
            is_fraud=prob >= THRESHOLD,
            threshold=THRESHOLD,
        )
    except KeyError as e:
        return jsonify(error=f"Missing field: {e}"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)