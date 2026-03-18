import os
import sys
import pickle

from src.classifier.exception import ClassifierException
from src.classifier.logger import logging
from dataclasses import dataclass

#  IMPORT SAME PREPROCESSING (VERY IMPORTANT)
from src.classifier.components.data_transformation import DataTransformation


# ================= CONFIG =================
@dataclass
class PredictionPipelineConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# ================= PIPELINE =================
class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = PredictionPipelineConfig.trained_model_file_path

            # 🔥 Load model + vectorizer
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.vectorizer = data["vectorizer"]

            # 🔥 SAME preprocessing as training
            self.preprocessor = DataTransformation()

            logging.info("Model and vectorizer loaded successfully")

        except Exception as e:
            raise ClassifierException(e, sys)

    def predict(self, text):
        try:
            # Ensure input is list
            if isinstance(text, str):
                text = [text]

            # 🔥 APPLY SAME CLEANING
            cleaned_text = [self.preprocessor.clean_text(t) for t in text]

            # Transform using TF-IDF
            X = self.vectorizer.transform(cleaned_text)

            # Prediction
            preds = self.model.predict(X)

            # 🔥 Convert to readable output
            results = ["Fake News" if p == 0 else "Real News" for p in preds]

            return results[0] if len(results) == 1 else results

        except Exception as e:
            raise ClassifierException(e, sys)