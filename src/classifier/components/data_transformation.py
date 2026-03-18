import os
import sys
import pandas as pd
import re

from dataclasses import dataclass
from src.classifier.exception import ClassifierException
from src.classifier.logger import logging

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ================= CONFIG =================
@dataclass
class DataTransformationConfig:
    transformed_data_path = os.path.join("data", "processed", "transformed_df.csv")


# ================= TRANSFORMATION =================
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # 🔹 TEXT CLEANING FUNCTION
    def clean_text(self, text):
        try:
            text = str(text)

            # Lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)

            # Remove punctuation
            text = re.sub(r"[^a-zA-Z\s]", "", text)

            # Remove extra spaces
            text = re.sub(r"\s+", " ", text).strip()

            # Remove stopwords
            words = text.split()
            words = [word for word in words if word not in ENGLISH_STOP_WORDS]

            return " ".join(words)

        except Exception as e:
            raise ClassifierException(e, sys)

    # 🔹 MAIN FUNCTION
    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("Starting Data Transformation")

            df = pd.read_csv(raw_data_path)

            # Select important columns (adjust if needed)
            if 'text' in df.columns:
                df = df[['text', 'label']]
            elif 'title' in df.columns:
                df = df[['title', 'label']]
                df.rename(columns={'title': 'text'}, inplace=True)

            # Rename for consistency
            df.columns = ['text', 'target']

            # Handle missing values
            df.dropna(inplace=True)

            # Apply cleaning
            df['transformed_text'] = df['text'].apply(self.clean_text)

            # Keep only required columns
            final_df = df[['transformed_text', 'target']]

            # Create directory
            os.makedirs(os.path.dirname(self.config.transformed_data_path), exist_ok=True)

            # Save
            final_df.to_csv(self.config.transformed_data_path, index=False)

            logging.info(f"Transformed data saved at {self.config.transformed_data_path}")

            return self.config.transformed_data_path

        except Exception as e:
            raise ClassifierException(e, sys)