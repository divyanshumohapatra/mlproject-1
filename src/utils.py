import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open (file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models:dict, params:dict):
    try:
        report = {}

        logging.info("Started calculating the r2 score for each model")
        logging.info("Preforming Hyperparameter tuning for the models")
        for model_name in models.keys():
            model = models[model_name]
            y_pred=[]
            if model_name in params:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid= params[model_name], 
                    scoring="r2",
                    cv=5,
                    n_jobs=-1       
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            report[model_name] = r2
        
        return report

    except Exception as e:
        raise CustomException(e, sys)