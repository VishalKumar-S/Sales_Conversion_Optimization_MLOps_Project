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
from steps.email_report import email_report
from sklearn.model_selection import train_test_split


@step(enable_cache=False)
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        preprocessor = DataPreprocessor(df)
        cleaned_data = preprocessor.clean_data()
        df=cleaned_data.copy()
        df.drop(['Approved_Conversion'], axis=1, inplace=True)
        # Splitting the data into reference and current datasets
        reference_dataset, current_dataset = train_test_split(df, train_size=70, random_state=42)
        return cleaned_data,reference_dataset, current_dataset
    except Exception as e:
        logging.error(e)
        raise e

@step(enable_cache=False)
def data_drift_validation(reference_dataset: pd.DataFrame, current_dataset: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    test_suite = TestSuite(tests=[DataDriftTestPreset(),])
    test_suite.run(reference_data=reference_dataset, current_data=current_dataset)
    threshold = test_suite.as_dict()['summary']['success_tests']/test_suite.as_dict()['summary']['total_tests']
    test_name = "Data Drift Test"
    passed_tests = test_suite.as_dict()['summary']['success_tests']
    failed_tests =test_suite.as_dict()['summary']['failed_tests']
    total_tests= test_suite.as_dict()['summary']['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted.")
    if(threshold< 0.65):
        test_suite.save_html("Reports/data_drift_suite.html")
        email_report(passed_tests, failed_tests, total_tests, "Data Drift Test", "Reports/data_drift_suite.html")
    else:
        logging.info(f"All Data Drift checks passed")
        return reference_dataset, current_dataset
        













