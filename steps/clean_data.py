import logging
import pandas as pd
from zenml import step
from typing import Tuple
from src.clean_data import DataPreprocessor
from typing import Tuple, Annotated
from steps.train_model import train_model
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import DataDriftTestPreset
from steps.alert_report import alert_report
from sklearn.model_selection import train_test_split
from neptune.types import File
import neptune
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
import streamlit as st
import sys


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        #Note: Here, the target column is removed since data drift validation is checked only for the input data, as in production, for batch dataset, only input data would be given, the target variable, we need to predict. Another reason is,including the target in drift analysis could give misleading results, as target shifts might be legitimate business changes rather than problematic data drift

        preprocessor = DataPreprocessor(df)
        cleaned_data = preprocessor.clean_data()
        cleaned_data_without_target= cleaned_data.copy()
        cleaned_data_without_target.drop(['Approved_Conversion'], axis=1, inplace=True)
        return cleaned_data, cleaned_data_without_target

class DataDriftValidator:
    @staticmethod
    def validate(reference_dataset: pd.DataFrame, current_dataset: pd.DataFrame, user_email: str)-> Tuple[pd.DataFrame, pd.DataFrame]:
        test_suite = TestSuite(tests=[DataDriftTestPreset(),])
        test_suite.run(reference_data= reference_dataset, current_data=current_dataset)
        threshold = test_suite.as_dict()['summary']['success_tests'] / test_suite.as_dict()['summary']['total_tests']
        test_name = "Data Drift Test"
        passed_tests = test_suite.as_dict()['summary']['success_tests']
        failed_tests = test_suite.as_dict()['summary']['failed_tests']
        total_tests = test_suite.as_dict()['summary']['total_tests']
        logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted in Data Drift.")
        st.write(f"Number of passed tests: {passed_tests} ✅, "
             f"Number of failed tests: {failed_tests} ❌, "
             f"Out of {total_tests} tests conducted in Data Drift.")


        if threshold < 0.65:
            logging.error("Data drift tests got failed. Logging failed reports and sending alerts...")
            neptune_run = get_neptune_run()

            test_suite.save_html("Evidently_Reports/data_drift_suite.html")
            neptune_run["html/Data Drift Test"].upload("Evidently_Reports/data_drift_suite.html")
            alert_report(passed_tests, failed_tests, total_tests, "Data Drift Test", "Evidently_Reports/data_drift_suite.html", user_email)
            sys.exit("Data drift threshold failed. Pipeline terminated.")
        else:
            logging.info(f"All Data Drift checks passed")
            return reference_dataset, current_dataset


@step(enable_cache=True)
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''By separating the ZenML pipeline steps(clean_data()) from the actual implementation logic(DataCleaner.clean()), I'm creating a clear boundary between orchestration and business logic. The step functions manage the workflow and error handling, while the class methods contain the actual data processing logic. Also,it allows for testing independednt components logic, and re-use the independent components logic'''
    try:
        cleaned_data = DataCleaner.clean(df)
        cleaned_data, cleaned_data_without_target = DataCleaner.clean(df)
        reference_dataset, current_dataset = train_test_split(cleaned_data_without_target, train_size=70, random_state=42)
        return cleaned_data, reference_dataset, current_dataset
    except Exception as e:
        logging.error(e)
        raise e



@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def data_drift_validation(reference_dataset: pd.DataFrame, current_dataset: pd.DataFrame, user_email: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        reference_dataset, current_dataset = DataDriftValidator.validate(reference_dataset, current_dataset, user_email)
        return reference_dataset, current_dataset
    except Exception as e:
        logging.error(e)
        raise e
