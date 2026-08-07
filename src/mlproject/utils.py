import pandas as pd
from dotenv import load_dotenv
import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

load_dotenv()  # Load environment variables from .env file
  # Get the CSV file path from environment variable
CSV_FILE = "D:/Project/MLOPS/notebook/StudentsPerformance.csv"
file_path = CSV_FILE  # Use the environment variable for the file path


def load_data():
    logging.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        return data
    
    except Exception as e:
        logging.info(f"Error loading data: {e}")
        raise CustomException(f"Error loading data: {e}", sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

            logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=2)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)