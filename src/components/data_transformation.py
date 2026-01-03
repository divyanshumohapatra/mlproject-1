import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Created pipeline for numerical features.")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oneHotEncoder", OneHotEncoder()),
                ]
            )

            logging.info("Created pipeline for categorical features.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Created the preprocessor.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_feature = "math_score"

            input_train_df = train_df.drop(columns=target_feature, axis=1)
            target_train_df = train_df[target_feature]

            input_test_df = test_df.drop(columns=target_feature, axis=1)
            target_test_df = test_df[target_feature]

            logging.info("Input and target data for training and testing purpose is dervied.")

            preprocessor = self.get_data_transformer()

            input_feature_train_arr = preprocessor.fit_transform(input_train_df)
            input_feature_test_arr = preprocessor.transform(input_test_df)

            train_arr = np.c_(
                input_feature_train_arr, np.array(target_train_df)
            )

            test_arr = np.c_(input_feature_test_arr, np.array(target_test_df))

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)