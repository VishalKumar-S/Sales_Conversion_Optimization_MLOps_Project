from zenml import step
from src.train_models import train_h2o_automl
import pandas as pd
import neptune
from zenml.integrations.neptune.experiment_trackers.run_state import (
    get_neptune_run
)
from zenml.client import Client
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import RegressionTestPreset
from steps.email_report import email_report
from sklearn.model_selection import train_test_split
from evidently import ColumnMapping
from typing import Tuple, Annotated
import logging

@step(experiment_tracker="neptune_experiment_tracker", enable_cache = False)
def train_model(cleaned_dataset: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize a run
    neptune_run = get_neptune_run()

    train_df, test_df = train_h2o_automl(cleaned_dataset)


    # Log parameters
    params = {
        "max_models": 20,
        "seed": 42,
    }
    neptune_run["parameters"] = params
    return train_df, test_df


@step(enable_cache=False)
def model_performance_validation(train_df : pd.DataFrame, test_df : pd.DataFrame):
    print(train_df.head(5))
    print(test_df.head(5))
    column_mapping = ColumnMapping()
    column_mapping.target = 'Approved_Conversion'
    column_mapping.prediction = 'prediction'
    test_suite = TestSuite(tests=[TestValueMeanError()])
    test_suite.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping)
    threshold = test_suite.as_dict()['summary']['success_tests']/test_suite.as_dict()['summary']['total_tests']
    test_name = "Model Performance Test"
    passed_tests = test_suite.as_dict()['summary']['success_tests']
    failed_tests =test_suite.as_dict()['summary']['failed_tests']
    total_tests= test_suite.as_dict()['summary']['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted.")
    if(threshold < 0.65):
        test_suite.save_html("Reports/model_performance_suite.html")
        email_report(passed_tests, failed_tests, total_tests, test_name, "Reports/model_performance_suite.html")
    else:
        logging.info(f"All Model Performance checks passed")




