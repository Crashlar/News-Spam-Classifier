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
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    Evaluate multiple models with GridSearchCV and return performance metrics.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models (dict): Dictionary of model name -> model object
        param (dict): Dictionary of model name -> parameter grid
    
    Returns:
        results (dict): Dictionary of model name -> metrics
    """

    results = {}

    for name, model in models.items():
        logging.info(f"Starting training for {name}...")

        try:
            # GridSearchCV for hyperparameter tuning
            grid = GridSearchCV(model, param[name], cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            logging.info(f"{name} best params: {grid.best_params_}")

            # Predictions
            y_pred = best_model.predict(X_test)

            # Metrics
            metrics = {
                "best_params": grid.best_params_,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            results[name] = metrics
            logging.info(f"{name} evaluation completed. F1 Score: {metrics['f1_score']:.4f}")

        except Exception as e:
            logging.error(f"Error while training {name}: {e}")
            

    return results
