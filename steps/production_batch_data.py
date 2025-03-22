from abc import ABC, abstractmethod
import pandas as pd
from zenml import step
import logging
import neptune
from neptune.types import File
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset
import streamlit as st
from steps.alert_report import alert_report
import sys

class DataReader:
    @staticmethod
    def read_data(data_path: str) -> pd.DataFrame:
        try:
            neptune_run = get_neptune_run()
            df = pd.read_csv(data_path)
            neptune_run["data/Prod_batch_data"].upload(File.as_html(df))
            logging.info("Read CSV file completed.")
            return df
        except Exception as e:
            logging.error(e)
            raise e



@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def production_batch_data(data_path: str) -> pd.DataFrame:
    try:
        return DataReader.read_data(data_path)
    except ValueError as ve:
        logging.error(ve)
        raise ve
    except Exception as e:
        logging.error(e)
        raise e



@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def data_quality_validation(ref_data: pd.DataFrame, curr_data: pd.DataFrame, user_email: str) -> pd.DataFrame:
    """ZenML Step: Validates data quality and triggers email on failure.
    Data quality focus on validating the intrinsic properties of the current batch of data i.e missing values, outliers, anamolies, data type consistency, duplicate records, Business rule violations, Format validation (dates, emails, phone numbers)"""
    test_suite = TestSuite(tests=[DataQualityTestPreset()])
    test_suite.run(reference_data= ref_data, current_data=curr_data)
    
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
        sys.exit("Data quality threshold failed on production data. Pipeline terminated.")
    else:
        return curr_data


@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def data_stability_validation(ref_data: pd.DataFrame, curr_data: pd.DataFrame, user_email: str) -> pd.DataFrame:
    """ZenML Step: Validates data stability and triggers email on failure.
    Data stability measures how data characteristics and distributions change over time by comparing the current dataset against historical reference data through tests for distribution shifts, statistical property changes, and feature correlation variations."""
    
    test_suite = TestSuite(tests=[DataStabilityTestPreset()])
    test_suite.run(reference_data= ref_data, current_data=curr_data)
    
    summary = test_suite.as_dict()['summary']
    passed_tests = summary['success_tests']
    failed_tests = summary['failed_tests']
    total_tests = summary['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted in Data Stability.")
    st.write(f"Number of passed tests: {passed_tests} ✅, "
             f"Number of failed tests: {failed_tests} ❌, "
             f"Out of {total_tests} tests conducted in Data Stability.")
        
    threshold = passed_tests / total_tests if total_tests > 0 else 0

    if threshold < 0.85:
        
        logging.error("Data stability tests got failed. Logging failed reports and sending alerts...")
        neptune_run = get_neptune_run()

        test_suite.save_html("Evidently_Reports/data_stability_suite.html")

        neptune_run["html/Data Stability Test"].upload("Evidently_Reports/data_stability_suite.html")
        
        alert_report(passed_tests, failed_tests, total_tests, "Data Stability Test", "Evidently_Reports/data_stability_suite.html", user_email)
        sys.exit("Data stability threshold failed. Pipeline terminated.")

    else:
        return curr_data