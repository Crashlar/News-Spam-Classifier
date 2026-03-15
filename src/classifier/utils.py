import os
import sys
from src.classifier.exception import ClassifierException
from src.classifier.logger import logging
import pandas as pd 
import kagglehub


import numpy as np



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

