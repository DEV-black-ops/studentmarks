import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from dataclasses import dataclass
import numpy as np
from src.mlproject.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    Attributes:
        preprocessor_obj_file_path (str): Path to save the preprocessor object.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        """
        Initializes the DataTransformation class with a DataTransformationConfig instance.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a data transformer object.

        Returns:
            ColumnTransformer: A column transformer object for data transformation.
        """
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]) 

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines created successfully.")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}") 

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully.")
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math score"
            numerical_columns = ["writing score", "reading score"]
            
            ##device train data in target and input features

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            ##device test data in target and input features
        
            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #combining the transformed input features with the target feature for both train and test datasets

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully.") 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)



