import pandas as pd
import requests
from io import StringIO
from typing import Optional
from zenml import step
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset
from steps.alert_report import alert_report
from neptune.types import File
import neptune
from neptune.types import File
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
from neptune.types import File
import neptune
import logging
from neptune.types import File
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
import streamlit as st
import os
import sys



@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
# Note: Here, i enabled caching, since we are retrieving data from a static dataset
def ingest_data(path: str) -> pd.DataFrame:
    """ZenML Step: Ingests data from the provided path"""
    return pd.read_csv(path)


@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def data_quality_validation(curr_data: pd.DataFrame, user_email: str) -> pd.DataFrame:
    """ZenML Step: Validates data quality and triggers email on failure"""
    test_suite = TestSuite(tests=[DataQualityTestPreset()])
    test_suite.run(reference_data=None, current_data=curr_data)
    
    summary = test_suite.as_dict()['summary']
    passed_tests = summary['success_tests']
    failed_tests = summary['failed_tests']
    total_tests = summary['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted in Data Quality.")
    st.write(f"Number of passed tests: {passed_tests} ✅, "
             f"Number of failed tests: {failed_tests} ❌, "
             f"Out of {total_tests} tests conducted in Data Quality.")
        
    threshold = passed_tests / total_tests if total_tests > 0 else 0

    if threshold < 0.85:

        logging.error("Data quality tests got failed. Logging failed reports and sending alerts...")
        # Initialize a run
        neptune_run = get_neptune_run()

        test_suite.save_html("Evidently_Reports/data_quality_suite.html")

        neptune_run["html/Data Quality Test"].upload("Evidently_Reports/data_quality_suite.html")
        
        alert_report(passed_tests, failed_tests, total_tests, "Data Quality Test", "Evidently_Reports/data_quality_suite.html", user_email)
        sys.exit("Data quality threshold failed. Pipeline terminated.")
    else:
        return curr_data