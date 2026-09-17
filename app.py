import os
import sys

from flask import Flask, jsonify, render_template, request

from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.exceptions import CustomException
from src.mlproject.logger import logging
from src.mlproject.pipelines.prediction_pipeline import PredictionPipeline

MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

application = Flask(__name__)
app = application


def train_model_if_needed():
    """Train the model only when required artifacts do not exist."""
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        logging.info("Model and preprocessor already exist. Skipping retraining.")
        return

    try:
        logging.info("Training artifacts not found. Starting model training...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed successfully.")

    except Exception as e:
        raise CustomException(e, sys)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            payload = request.get_json(force=True)
        else:
            payload = {
                "gender": request.form.get("gender"),
                "race/ethnicity": request.form.get("race/ethnicity"),
                "parental level of education": request.form.get("parental level of education"),
                "lunch": request.form.get("lunch"),
                "test preparation course": request.form.get("test preparation course"),
                "writing score": float(request.form.get("writing score")),
                "reading score": float(request.form.get("reading score")),
            }

        if not payload:
            return jsonify({"error": "Request body must be a JSON object or a form submission."}), 400

        prediction_pipeline = PredictionPipeline(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)
        prediction = prediction_pipeline.predict(payload)

        if request.is_json:
            return jsonify({
                "prediction": float(prediction),
                "unit": "math_score"
            })

        return render_template(
            'index.html',
            prediction=float(prediction),
            form_data=payload
        )

    except ValueError as ve:
        if request.is_json:
            return jsonify({"error": str(ve)}), 400
        return render_template('index.html', error=str(ve))
    except FileNotFoundError as fnf:
        if request.is_json:
            return jsonify({"error": f"Model artifacts missing: {str(fnf)}"}), 500
        return render_template('index.html', error=f"Model artifacts missing: {str(fnf)}")
    except Exception as e:
        logging.exception("Prediction request failed")
        if request.is_json:
            return jsonify({"error": "Prediction failed", "details": str(e)}), 500
        return render_template('index.html', error="Prediction failed. Please check inputs.")


if __name__ == "__main__":
    train_model_if_needed()
    app.run(host="0.0.0.0", port=5000, debug=True) 