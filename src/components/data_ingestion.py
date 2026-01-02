import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv") 
    test_data_path:str = os.path.join("artifacts", "test.csv") 
    raw_data_path:str = os.path.join("artifacts", "raw.csv") 


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Started Data Ingestion component")
        try:
            df = pd.read_csv("src/notebook/data/stud.csv")
            logging.info("Read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Perform Train test split")

            train_data, test_data = train_test_split(df, test_size=0.25, random_state=43)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            
        except Exception as e:
            raise CustomException(e, sys)