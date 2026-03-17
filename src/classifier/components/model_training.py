import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.classifier.exception import ClassifierException
from src.classifier.logger import logging
from src.classifier.utils import evaluate_models, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


# ================= CONFIG =================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    transformed_df_path = os.path.join("data", "processed", "transformed_df.csv")


# ================= TRAINER =================
class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()

    def split_data(self, df):
        try:
            logging.info("Cleaning data + TF-IDF")

            #  AGAIN DOING CLEANING DUE TO RE CHECK
            df = df[['transformed_text', 'target']]
            df = df.dropna(subset=['transformed_text'])
            df['transformed_text'] = df['transformed_text'].astype(str)

            #  TF-IDF
            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(df['transformed_text'])
            y = df['target'].values

            #  SPLIT
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )

            return X_train, X_test, y_train, y_test, tfidf

        except Exception as e:
            raise ClassifierException(e, sys)

    def intiate_model_trainer(self):
        try:
            # ================= LOAD DATA =================
            logging.info("Loading transformed dataset")
            df = pd.read_csv(self.model_training_config.transformed_df_path)

            # ================= SPLIT =================
            X_train, X_test, y_train, y_test, tfidf = self.split_data(df)

            # ================= MODELS =================
            models = {
                "SVC": SVC(),
                "KNN": KNeighborsClassifier(),
                "MultinomialNB": MultinomialNB(),
                "DecisionTree": DecisionTreeClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "ExtraTrees": ExtraTreesClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            # ================= HYPERPARAMETERS =================
            params = {
                "SVC": {
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 10]
                },
                "KNN": {
                    "n_neighbors": [3, 5],
                },
                "MultinomialNB": {
                    "alpha": [0.5, 1.0]
                },
                "DecisionTree": {
                    "max_depth": [5, 10]
                },
                "LogisticRegression": {
                    "C": [0.1, 1, 10]
                },
                "RandomForest": {
                    "n_estimators": [50],
                },
                "AdaBoost": {
                    "n_estimators": [50],
                },
                "Bagging": {
                    "n_estimators": [50],
                },
                "ExtraTrees": {
                    "n_estimators": [50],
                },
                "GradientBoosting": {
                    "n_estimators": [50],
                },
                "XGBoost": {
                    "n_estimators": [50],
                    "max_depth": [3, 5]
                }
            }

            # ================= TRAIN =================
            logging.info("Starting model evaluation")

            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # ================= BEST MODEL =================
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            print(f"Best Model: {best_model_name}")
            print(f"Best F1 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise ClassifierException("No good model found", sys)

            # ================= SAVE =================
            logging.info("Saving best model and TF-IDF vectorizer")

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj={"model": best_model, "vectorizer": tfidf}  # ✅ IMPORTANT
            )

            # ================= FINAL ACCURACY =================
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return accuracy

        except Exception as e:
            raise ClassifierException(e, sys)