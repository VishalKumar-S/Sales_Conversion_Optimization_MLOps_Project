from zenml import step
import pandas as pd
import neptune
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
from evidently import ColumnMapping
from typing import Tuple
from evidently.test_suite import TestSuite
from evidently.tests import TestValueMeanError
from steps.email_report import email_report
import logging

@step(experiment_tracker="neptune_experiment_tracker", enable_cache=False)
def train_model(cleaned_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize a run
    neptune_run = get_neptune_run()

    # Your existing train_h2o_automl logic here
    train_df, test_df = train_h2o_automl(cleaned_dataset)

    # Log parameters to Neptune
    params = {
        "max_models": 20,
        "seed": 42,
    }
    neptune_run["parameters"] = params

    return train_df, test_df

@step(enable_cache=False)
def model_performance_validation(train_df: pd.DataFrame, test_df: pd.DataFrame):
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
    
    if threshold < 0.65:
        test_suite.save_html("Reports/model_performance_suite.html")
        email_report(passed_tests, failed_tests, total_tests, test_name, "Reports/model_performance_suite.html")
    else:
        logging.info(f"All Model Performance checks passed")
