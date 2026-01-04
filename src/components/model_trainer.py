import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Starting the model training")
            logging.info("Started the splitting of training and testing data into dependent and independent structure.")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "SVR": SVR(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "XgBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor()
            }

            params = {
                "SVR":{
                    "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
                    "C":[10, 1, 0.1, 0.01],
                },
                "DecisionTree":{
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter": ['best', 'random'],
                    "max_depth":[2, 5, 10, 25, 50],
                    "max_features": ['sqrt', 'log2', None]
                },
                "RandomForest":{
                    "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    "max_features": ['sqrt', 'log2'],
                    "max_depth": [2, 5, 10, 25, 50]
                },
                "AdaBoost":{
                    "n_estimators": [50, 100, 150],
                    "loss":['linear', 'square', 'exponential']
                },
                "GradientBoost":{
                    "loss":['squared_error', 'absolute_error', 'huber', 'quantile'],
                    "n_estimators": [50, 100, 150],
                    "criterion":['friedman_mse', 'squared_error']
                }
            }
            report = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            max_score = max(sorted(report.values()))
            best_model_name = ""

            for model_name in report.keys():
                if report[model_name] == max_score:
                    best_model_name = model_name
                    break
            
            print(f"The best model is {best_model_name} with r2_score {max_score}")

            logging.info("The best model is found.")

            best_model = models[best_model_name]

            save_object(
                file_path=  self.model_trainer_config.model_trainer_path,
                obj=best_model
            )
            
        except Exception as e:
            raise CustomException(e, sys)
