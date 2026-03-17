import os 
import sys 
from src.classifier.exception import ClassifierException
from src.classifier.logger import logging
from dataclasses import dataclass
import pickle

@dataclass
class PredictionPipelineConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    transformed_df_path = os.path.join("data", "processed", "transformed_df.csv")
class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = PredictionPipelineConfig.trained_model_file_path

            # load model and vectorization 
            with open(self.model_path , "rb") as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.vectorizer = data['vectorizer']

            logging.info("model and vectorizer loaded successfully")

        except Exception as e:
            raise ClassifierException(e , sys)
        
    def predict(self , text):
        try:
            # basic preprocesssing as same as training 
            if isinstance(text , str):
                text = [text]
            text = [t.strip().lower() for t in text]

            # transform using saved TF-IDF
            X = self.vectorizer.transform(text)

            # preiction
            preds = self.model.predict(X)

            return preds
        except Exception as e:
            raise ClassifierException(e , sys)
        