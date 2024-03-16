import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


@dataclass
class DataTrnasfromationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')


class DataTrnasformation:
    def __init__(self):
        self.data_tranasformation_config = DataTrnasfromationConfig()

    def get_trnasformer_object(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Numerical columns standard scaling completed')
            logging.info('Categorical columns encoding comleted')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_trnasformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test dat completed')
            logging.info('Obtaining preprocessing Object')

            preprocessing_obj = self.get_trnasformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score','reading_score']\
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_arr)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('saved preprocessing object')

            save_object(filepath = self.data_tranasformation_config.preprocessor_obj_file_path,obj = preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_tranasformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
