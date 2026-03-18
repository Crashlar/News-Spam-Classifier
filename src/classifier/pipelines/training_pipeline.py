import sys
from src.classifier.exception import ClassifierException
from src.classifier.logger import logging

# Import your components
from src.classifier.components.data_ingestion import DataIngestion
from src.classifier.components.data_transformation import DataTransformation
from src.classifier.components.model_training import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("🚀 Training Pipeline Started")

            # ================= STEP 1: DATA INGESTION =================
            logging.info("📥 Starting Data Ingestion")
            data_ingestion = DataIngestion()
            raw_data_path = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion Completed. File saved at: {raw_data_path}")

            # ================= STEP 2: DATA TRANSFORMATION =================
            logging.info("🔄 Starting Data Transformation")
            data_transformation = DataTransformation()
            transformed_data_path = data_transformation.initiate_data_transformation(raw_data_path)

            logging.info(f"Data Transformation Completed. File saved at: {transformed_data_path}")

            # ================= STEP 3: MODEL TRAINING =================
            logging.info("🤖 Starting Model Training")
            model_trainer = ModelTrainer()
            accuracy = model_trainer.intiate_model_trainer()

            logging.info(f"Model Training Completed. Accuracy: {accuracy}")

            logging.info("✅ Training Pipeline Completed Successfully")

            return accuracy

        except Exception as e:
            raise ClassifierException(e, sys)