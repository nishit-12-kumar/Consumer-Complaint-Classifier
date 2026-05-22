import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml_config

@dataclass
class DataIngestionConfig:
    # We can load these from config.yaml, but dataclass provides a clean fallback/structure
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # Optionally load dynamic variables from config.yaml
        self.config = read_yaml_config("config.yaml")

    def initiate_data_ingestion(self):
        logging.info("Initiating Data Ingestion Phase")
        try:
            raw_data_path = self.ingestion_config.raw_data_path
            
            if not os.path.exists(raw_data_path):
                raise FileNotFoundError(f"Dataset not found at {raw_data_path}. Please place your raw data here.")

            # Read the dataset
            df = pd.read_csv(raw_data_path)
            logging.info(f"Successfully read the dataset as dataframe. Shape: {df.shape}")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Perform Train-Test Split
            logging.info("Initiating train-test split")
            test_size = self.config.get("data_ingestion", {}).get("test_size", 0.2)
            random_state = self.config.get("data_ingestion", {}).get("random_state", 42)
            
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)

            # Save the split datasets back to the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion phase completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # For testing the ingestion component in isolation
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved at: {train_data}")
    print(f"Test data saved at: {test_data}")