from src.mlproject.logger import logging
import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.exceptions import CustomException
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    logging.info("Starting the ML project setup...")
    # Additional setup code can be added here
    

    try:
        ## data_ingestion        
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        ## data_transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        ## model_trainer
        model_trainer = ModelTrainer()
        print(model_trainer.initate_model_trainer(train_arr,test_arr))


    except Exception as e:
        logging.info("An error occurred during the data ingestion process.")
        raise CustomException(e, sys) 