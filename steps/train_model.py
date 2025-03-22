from zenml import step
import pandas as pd
import neptune
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
from evidently import ColumnMapping
from typing import Tuple
from evidently.test_suite import TestSuite
from evidently.tests import TestValueMeanError
from steps.alert_report import alert_report
import logging
from neptune.types import File
import neptune
import streamlit as st
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
from src.train_models import train_h2o_automl
import sys

@step(experiment_tracker="neptune_experiment_tracker", enable_cache=True)
def train_model(cleaned_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    neptune_run = get_neptune_run()


    train_df, test_df = train_h2o_automl(cleaned_dataset)

    # Log H2o parameters to Neptune
    params = {
        "max_models": 20,
        "seed": 42,
    }
    neptune_run["parameters"] = params

    return train_df, test_df

@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def model_performance_validation(train_df: pd.DataFrame, test_df: pd.DataFrame, user_email: str):

    column_mapping = ColumnMapping()
    column_mapping.target = 'Approved_Conversion'
    column_mapping.prediction = 'prediction'
    test_suite = TestSuite(tests=[TestValueMeanError()])
    test_suite.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping)
    threshold = test_suite.as_dict()['summary']['success_tests'] / test_suite.as_dict()['summary']['total_tests']
    test_name = "Model Performance Test"
    passed_tests = test_suite.as_dict()['summary']['success_tests']
    failed_tests = test_suite.as_dict()['summary']['failed_tests']
    total_tests = test_suite.as_dict()['summary']['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted.")
    st.write(f"Number of passed tests: {passed_tests} ✅, "
             f"Number of failed tests: {failed_tests} ❌, "
             f"Out of {total_tests} tests conducted in model validation.")

    if threshold < 0.65:

        logging.error("Model performance tests got failed. Logging failed reports and sending alerts...")
        # Initialize a neptune run
        neptune_run = get_neptune_run()

        test_suite.save_html("Evidently_Reports/model_performance_suite.html")
        neptune_run["html/Model Performance"].upload("Evidently_Reports/model_performance_suite.html")
        alert_report(passed_tests, failed_tests, total_tests, test_name, "Evidently_Reports/model_performance_suite.html", user_email)
        sys.exit("Model performance threshold failed. Pipeline terminated.")
    else:
        logging.info(f"All Model Performance checks passed")
