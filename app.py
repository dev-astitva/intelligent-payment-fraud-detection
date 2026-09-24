from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

app = Flask(__name__)

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run notebooks/fraud_detection_analysis.ipynb "
            "first to train and save models/fraud_model.pkl."
        )
    return joblib.load(MODEL_PATH)


try:
    ARTIFACT = load_artifact()
    MODEL = ARTIFACT["pipeline"]
    DECISION_THRESHOLD = float(ARTIFACT["threshold"])
    RISK_THRESHOLDS = ARTIFACT.get(
        "risk_thresholds",
        {"low_max": 0.30, "medium_max": 0.70},
    )
except Exception as exc:
    ARTIFACT = None
    MODEL = None
    DECISION_THRESHOLD = 0.50
    RISK_THRESHOLDS = {"low_max": 0.30, "medium_max": 0.70}
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


def get_risk_level(probability):
    if probability < RISK_THRESHOLDS["low_max"]:
        return "LOW"
    if probability < RISK_THRESHOLDS["medium_max"]:
        return "MEDIUM"
    return "HIGH"


def get_prediction_label(probability):
    return (
        "Potentially Fraudulent"
        if probability >= DECISION_THRESHOLD
        else "Likely Legitimate"
    )


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        transaction_types=TRANSACTION_TYPES,
        result=None,
        error=MODEL_LOAD_ERROR,
        model_name=ARTIFACT.get("model_name") if ARTIFACT else None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template(
            "index.html",
            transaction_types=TRANSACTION_TYPES,
            result=None,
            error=(
                "The trained model is not available. Run the notebook first "
                "to generate models/fraud_model.pkl."
            ),
            model_name=None,
        ), 500

    try:
        step = int(request.form.get("step", ""))
        transaction_type = request.form.get("type", "")
        amount = float(request.form.get("amount", ""))

        if step < 1 or step > 744:
            raise ValueError("Step must be between 1 and 744.")
        if transaction_type not in TRANSACTION_TYPES:
            raise ValueError("Please select a valid transaction type.")
        if amount < 0:
            raise ValueError("Amount cannot be negative.")

        input_data = pd.DataFrame(
            [{
                "step": step,
                "type": transaction_type,
                "amount": amount,
            }]
        )

        probability = float(MODEL.predict_proba(input_data)[0, 1])
        prediction = get_prediction_label(probability)
        risk = get_risk_level(probability)

        result = {
            "probability": probability * 100,
            "prediction": prediction,
            "risk": risk,
            "threshold": DECISION_THRESHOLD * 100,
        }

        return render_template(
            "index.html",
            transaction_types=TRANSACTION_TYPES,
            result=result,
            error=None,
            model_name=ARTIFACT.get("model_name"),
            form_data={
                "step": step,
                "type": transaction_type,
                "amount": amount,
            },
        )

    except (ValueError, TypeError):
        error = (
            "Please enter a valid step (1–744), choose a transaction type, "
            "and enter a non-negative amount."
        )
        return render_template(
            "index.html",
            transaction_types=TRANSACTION_TYPES,
            result=None,
            error=error,
            model_name=ARTIFACT.get("model_name"),
            form_data=request.form,
        ), 400
    except Exception:
        error = "The transaction could not be analyzed. Please check the inputs and try again."
        return render_template(
            "index.html",
            transaction_types=TRANSACTION_TYPES,
            result=None,
            error=error,
            model_name=ARTIFACT.get("model_name"),
            form_data=request.form,
        ), 500


if __name__ == "__main__":
    app.run(debug=False)
