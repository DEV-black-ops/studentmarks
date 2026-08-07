import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
from src.mlproject.exceptions import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from src.mlproject.utils import load_data




@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    Attributes:
        train_data_path (str): Path to save the training data.
        test_data_path (str): Path to save the testing data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion class with a DataIngestionConfig instance.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process.")
            # Read the CSV file
            data = load_data()
            logging.info("Data read successfully.")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")
            # Split the data into training and testing sets

            train_set, test_set = train_test_split(data, test_size=self.ingestion_config.test_size, random_state=self.ingestion_config.random_state)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")
            logging.info("Data ingestion process completed successfully.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(f"Error creating directories: {e}", sys)    

        