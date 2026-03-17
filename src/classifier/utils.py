import os
import sys
from src.classifier.exception import ClassifierException
from src.classifier.logger import logging

import pandas as pd 
import kagglehub
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def fetch_data():
    """
    Loads all CSV files from Kaggle dataset.
    Returns a dictionary {filename: dataframe}.
    """
    try:
        # Download dataset from Kaggle
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        logging.info(f"Dataset downloaded at: {path}")

        dataframes = {}

        # Loop through all CSV files in the dataset folder
        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path, low_memory=False)  # low_memory avoids dtype warnings
                dataframes[file] = df
                logging.info(f"Loaded {file} with shape {df.shape}")

        if not dataframes:
            raise FileNotFoundError("No CSV files found in downloaded dataset")

        return dataframes

    except Exception as e:
        raise ClassifierException(e, sys)
    
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise ClassifierException(e, sys)
 
 
def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    results = {}
    trained_models = {}

    for name, model in models.items():
        logging.info(f"Starting training for {name}...")

        try:
            grid = GridSearchCV(model, param[name], cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            trained_models[name] = best_model  

            logging.info(f"{name} best params: {grid.best_params_}")

            y_pred = best_model.predict(X_test)

            f1 = f1_score(y_test, y_pred)

            results[name] = f1  
            
            logging.info(f"{name} F1 Score: {f1:.4f}")

        except Exception as e:
            logging.error(f"Error while training {name}: {e}")

    return results, trained_models