from src.classifier.logger import logging
from src.classifier.exception import ClassifierException
from src.classifier.components.model_training import  ModelTrainer , ModelTrainerConfig
from src.classifier.components import DataIngestionConfig , DataIngestion
import sys


if __name__ == "__main__":
    logging.info("The application has started ")
    try:
        # Run data ingestion
        data_ingestion = DataIngestion()
        saved_files = data_ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion completed successfully")
        print("Saved files:", saved_files)
    except Exception as e:
        logging.info("Custom Exception during Data ingestion")
        raise ClassifierException(e , sys)