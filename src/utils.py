import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score

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
    
def evaluate_models(X_train, X_test, y_train, y_test, models:dict):
    try:
        report = {}

        logging.info("Started calculating the r2 score for each model")
        for model_name in models.keys():
            model = models[model_name]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            report[model_name] = r2
        
        return report

    except Exception as e:
        raise CustomException(e, sys)