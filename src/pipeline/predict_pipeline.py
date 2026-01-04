import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict_data(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            features = preprocessor.transform(features)
            pred = model.predict(features)
            return pred
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 gender:str,
                 ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:int,
                 reading_score:int,
                 writing_score:int):
        self.gender = gender
        self.ethnicity = ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_custom_data(self):
        try:
            data = {
                "gender":self.gender,
                "race_ethnicity":self.ethnicity,
                "parental_level_of_education":self.parental_level_of_education,
                "lunch":self.lunch,
                "test_preparation_course":self.test_preparation_course,
                "reading_score":self.reading_score,
                "writing_score":self.writing_score
            }

            return pd.DataFrame(data, index=[0])
        except Exception as e:
            raise CustomException(e, sys)
        
        
        