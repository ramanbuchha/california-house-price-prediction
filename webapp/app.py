from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

reg_model = joblib.load(os.path.join(MODEL_DIR, "regression_model.pkl"))
cls_model = joblib.load(os.path.join(MODEL_DIR, "classification_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
thresholds = joblib.load(os.path.join(MODEL_DIR, "thresholds.pkl"))
cluster_model = joblib.load(os.path.join(MODEL_DIR, "clustering_model.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Create DataFrame with correct feature names
        input_df = {
            "MedInc": float(request.form["MedInc"]),
            "HouseAge": float(request.form["HouseAge"]),
            "AveRooms": float(request.form["AveRooms"]),
            "AveBedrms": float(request.form["AveBedrms"]),
            "Population": float(request.form["Population"]),
            "AveOccup": float(request.form["AveOccup"]),
            "Latitude": float(request.form["Latitude"]),
            "Longitude": float(request.form["Longitude"])
        }

        import pandas as pd
        input_df = pd.DataFrame([input_df])

        # Scale
        input_scaled = scaler.transform(input_df)

        # Regression prediction
        reg_prediction = reg_model.predict(input_scaled)[0]

        # Classification prediction
        cls_prediction = cls_model.predict(input_scaled)[0]

        class_labels = {
            0: "Low Value",
            1: "Medium Value",
            2: "High Value"
        }

        # Clustering prediction
        cluster_prediction = cluster_model.predict(input_scaled)[0]

        cluster_labels = {
            0: "Cluster 0 (Region Type A)",
            1: "Cluster 1 (Region Type B)",
            2: "Cluster 2 (Region Type C)"
        }


        return render_template(
            "index.html",
            regression_result=round(reg_prediction, 3),
            classification_result=class_labels[cls_prediction],
            cluster_result=cluster_labels[cluster_prediction]
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
