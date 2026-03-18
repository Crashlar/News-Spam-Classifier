from src.classifier.logger import logging
from src.classifier.exception import ClassifierException
from src.classifier.components.model_training import  ModelTrainer , ModelTrainerConfig
from src.classifier.components import DataIngestionConfig , DataIngestion
import sys

if __name__ == "__main__":
    logging.info("The application has started ")
    try:
        model_trainer = ModelTrainer()
        print(model_trainer.intiate_model_trainer())
    except Exception as e:
        logging.info("Custom Exception during Data ingestion")
        raise ClassifierException(e , sys)


# from src.classifier.pipelines.prediction_pipeline import PredictionPipeline

# try:
#     predictor = PredictionPipeline()

#     text = "Breaking news! Goverment announces new policy"

#     result  = predictor.predict(text)

#     print("Prediction : ", result)

# except Exception as e:
#     print(e)