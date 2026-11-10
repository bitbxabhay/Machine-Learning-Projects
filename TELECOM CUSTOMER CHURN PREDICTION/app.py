import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE)      # index.html isi folder me hai

def find_value(d, *words):
    """dict me aisi key dhoondo jisme koi bhi word aata ho (jaise 'thresh', 'auc')."""
    for k, v in d.items():
        if any(w in str(k).lower() for w in words) and isinstance(v, (int, float)):
            return float(v)
    return None


saved = joblib.load(os.path.join(BASE, "gb_model_churn.pkl"))
THRESHOLD, ROC_AUC = None, None

if isinstance(saved, dict):
    # model + threshold + roc value ek dict me saved hai
    model = next(v for v in saved.values() if hasattr(v, "predict_proba"))
    THRESHOLD = find_value(saved, "thresh")
    ROC_AUC = find_value(saved, "auc", "roc")
else:
    model = saved
    THRESHOLD = getattr(saved, "threshold", None)
    ROC_AUC = getattr(saved, "roc_auc", None)

if THRESHOLD is None:
    THRESHOLD = 0.5          # model file me threshold nahi mila to default
if ROC_AUC is not None:
    ROC_AUC = round(ROC_AUC, 3)
else:
    ROC_AUC = "N/A"

print("Loaded -> threshold:", THRESHOLD, "| roc_auc:", ROC_AUC)
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
train_columns = list(joblib.load(os.path.join(BASE, "columns.pkl")))

# scaler kin columns par fit hua tha
if hasattr(scaler, "feature_names_in_"):
    SCALE_COLS = list(scaler.feature_names_in_)
else:
    SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
TOGGLES = ["SeniorCitizen", "Partner", "Dependents"]
NET_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]


def preprocess(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row], columns=FEATURES)

    # text columns -> one-hot, phir bilkul training wale columns jaisa bana do
    df = pd.get_dummies(df)
    df = df.reindex(columns=train_columns, fill_value=0)

    # numeric columns scale karo
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df.astype(float)


@app.route("/", methods=["GET", "POST"])
def home():
    form, result, error = {}, None, None

    if request.method == "POST":
        form = request.form.to_dict()

        if form.get("PhoneService") == "No":
            form["MultipleLines"] = "No phone service"
        if form.get("InternetService") == "No":
            for name in NET_SERVICES:
                form[name] = "No internet service"

        try:
            row = {c: form.get(c) for c in FEATURES}
            for t in TOGGLES:
                row[t] = "Yes" if form.get(t) else "No"
            row["SeniorCitizen"] = 1 if form.get("SeniorCitizen") else 0
            row["tenure"] = int(float(row["tenure"]))
            row["MonthlyCharges"] = float(row["MonthlyCharges"])
            row["TotalCharges"] = float(row["TotalCharges"])

            X = preprocess(row)
            prob = float(model.predict_proba(X)[0][1])

            churn = prob >= THRESHOLD
            result = {
                "probability": round(prob * 100, 1),
                "churn": churn,
                "label": "Likely to Churn" if churn else "Likely to Stay",
                "message": "This customer may leave the service." if churn
                else "This customer is likely to continue the service.",
            }
        except Exception as e:
            error = f"Could not get a prediction: {e}"

    return render_template("index.html", form=form, result=result, error=error,
                           threshold=THRESHOLD, roc_auc=ROC_AUC)


if __name__ == "__main__":
    app.run(debug=True)