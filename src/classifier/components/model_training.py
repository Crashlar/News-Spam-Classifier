from src.classifier.exception import ClassifierException
from src.classifier.logger import logging
from src.classifier.utils import evaluate_models , save_object


import pandas as pd 
import numpy as np 
import os
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , BaggingClassifier, ExtraTreesClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    transformed_df_path = os.path.join("data", "processed", "transformed_df.csv")
    

class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()
        
    
    def eval_metrics(self, actual, pred, proba=None):
        """
        Evaluate classification metrics.
        actual: true labels
        pred: predicted labels
        proba: predicted probabilities (optional, for ROC-AUC)
        """
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="weighted")
        recall = recall_score(actual, pred, average="weighted")
        f1 = f1_score(actual, pred, average="weighted")
        cm = confusion_matrix(actual, pred)
        
        auc = None
        if proba is not None:
            try:
                auc = roc_auc_score(actual, proba, multi_class="ovr")
            except Exception as e:
                auc = str(e)
        
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "roc_auc": auc
        }
        

    def split_data(self , df):
        """
        data_path
        """
        try:
            logging.info("Split training and test input data")
            
            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform(df['transformed_text']).toarray()
            y = df['target'].values
            
            # split the data 
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
            return (
                X_train,
                X_test,
                y_train,
                y_test
            )
        except Exception as ey:
            raise ClassifierException(ey , sys)
        
        
        
    def intiate_model_trainer(self):
        try:
            # split the data  using function 
            X_train, X_test, y_train, y_test = self.split_data(self.model_training_config.transformed_df_path)
            
            
            models = {
                "SVC": SVC(kernel='sigmoid', gamma=1.0),
                "KNN": KNeighborsClassifier(),
                "MultinomialNB": MultinomialNB(),
                "DecisionTree": DecisionTreeClassifier(max_depth=5),
                "LogisticRegression": LogisticRegression(solver='liblinear', penalty='l1'),
                "RandomForest": RandomForestClassifier(n_estimators=50, random_state=2),
                "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=2),
                "Bagging": BaggingClassifier(n_estimators=50, random_state=2),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=50, random_state=2),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=2),
                "XGBoost": XGBClassifier(n_estimators=50, random_state=2)
            }

            # Params for HyperParameter Tuning 
            params = {
                "SVC": {
                    "kernel":["sigmoid", "linear", "rbf"],
                    "gamma": [0.1, 1.0, 10]
                    },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                    },
                "MultinomialNB": {
                    "alpha": [0.1, 0.5, 1.0]
                    },
                "DecisionTree": {
                    "max_depth": [3, 5, 10],
                    "criterion": ["gini", "entropy"]
                    },
                "LogisticRegression": {
                    "solver": ["liblinear"], 
                    "penalty": ["l1", "l2"]
                    },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                 "learning_rate": [0.5, 1.0]
                },
                "Bagging": {
                    "n_estimators": [50, 100]
                ,   "max_samples": [0.5, 1.0]
                },
                "ExtraTrees": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10]
                },
                "GradientBoosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 0.2]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5, 7]
                }
            }
            # Train the model 
            model_report :dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)

            # check performance 
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)
            
            if best_model_score<0.6:
                raise ClassifierException("No best model found" , sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy_ = accuracy_score(y_test, predicted)
            return accuracy_
            
            # save the model 
             
        except Exception as ex:
            raise ClassifierException(ex, sys)         
