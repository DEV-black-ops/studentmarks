# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
from dataclasses import dataclass
import os
import sys
from src.mlproject.utils import save_object, evaluate_models
import mlflow
from urllib.parse import urlparse



@dataclass
class ModelTrainerConfig():

    model_trainer_path = os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self) :
        self.model_trainer_config =ModelTrainerConfig()


    def eval_metrics(self,true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square


    def initate_model_trainer(self,train_array,test_array):
        try:
            logging.info("initating train test split")
            X_train,Y_train,X_test,Y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models = {
                        "Linear Regression": LinearRegression(),
                        
                        "Decision Tree": DecisionTreeRegressor(),
                        
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                        }

            ## we can add parameters for hyperparameter tuning for each model 
            #params = {}

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("traning all the models")

            model_report:dict=evaluate_models(X_train,Y_train,X_test,Y_test,models,params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            ### ml flow for expriment tracking 



            model_names = list(params.keys())

            actual_model = ""
            for model in model_names:
               if model == best_model_name:
                   actual_model = actual_model + model

            best_params = params[actual_model]

            # Configure MLflow to use Dagshub if DAGSHUB_REPO is provided (format: owner/repo)
            dagshub_repo = os.getenv("DAGSHUB_REPO")
            dagshub_token = os.getenv("DAGSHUB_TOKEN")

            if dagshub_repo:
                tracking_uri = f"https://dagshub.com/{dagshub_repo}.mlflow"
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_registry_uri(tracking_uri)
                    logging.info(f"MLflow tracking and registry URI set to {tracking_uri}")

                    # If a token is provided, export common MLflow auth env vars so MLflow can authenticate to Dagshub
                    if dagshub_token:
                        # Set multiple possible env vars to increase compatibility across MLflow versions
                        os.environ.setdefault("MLFLOW_TRACKING_TOKEN", dagshub_token)
                        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "")
                        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)
                        logging.info("MLflow Dagshub token set from DAGSHUB_TOKEN environment variable")
                except Exception as uri_error:
                    logging.info(f"Failed to set Dagshub tracking URI: {uri_error}")
            else:
                logging.info(f"DAGSHUB_REPO not set. Using existing MLflow tracking URI: {mlflow.get_tracking_uri()}")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Start MLflow run and log metrics/params/model
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(Y_test, predicted_qualities)

                # log hyperparameters if available
                try:
                    mlflow.log_params(best_params)
                except Exception:
                    # best_params may be empty or not serializable
                    logging.info("Best params not logged (empty or not serializable)")

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry may not be available depending on tracking backend and permissions
                if tracking_url_type_store != "file":
                    try:
                        # Attempt to register the model in the remote MLflow registry (MLflow 3.x: use name and registered_model_name)
                        mlflow.sklearn.log_model(best_model, name="model", registered_model_name=actual_model)
                        logging.info(f"Model logged and registered as '{actual_model}' in remote registry")
                    except Exception as registry_error:
                        logging.info(f"MLflow registry unavailable or permission denied: {registry_error}")
                        logging.info("Falling back to logging the model without registry registration.")
                        mlflow.sklearn.log_model(best_model, name="model")
                else:
                    mlflow.sklearn.log_model(best_model, name="model")

            ###### ml flow from code end .........

            print("This is the best model:")
            print(best_model_name)
            save_object(
                            file_path=self.model_trainer_config.model_trainer_path,
                            obj=best_model
                        )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)
            return r2_square


        except Exception as e:
            return CustomException(e,sys)




