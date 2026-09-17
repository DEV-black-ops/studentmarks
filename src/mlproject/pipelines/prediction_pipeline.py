import os
import pickle

import pandas as pd


class PredictionPipeline:
    """Load the trained model and preprocessor and predict the target value."""

    REQUIRED_FEATURES = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "writing score",
        "reading score"
    ]

    def __init__(self, model_path, preprocessor_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

        with open(self.model_path, "rb") as model_file:
            self.model = pickle.load(model_file)

        with open(self.preprocessor_path, "rb") as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)

    def predict(self, input_data):
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary with feature names as keys.")

        missing_features = [feature for feature in self.REQUIRED_FEATURES if feature not in input_data]
        if missing_features:
            raise ValueError(f"Missing required fields: {missing_features}")

        feature_df = pd.DataFrame([input_data])
        feature_df = feature_df[self.REQUIRED_FEATURES]

        transformed_features = self.preprocessor.transform(feature_df)
        prediction = self.model.predict(transformed_features)
        return prediction[0]
