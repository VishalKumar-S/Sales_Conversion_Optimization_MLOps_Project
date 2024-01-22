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

class DataFetcher:
    """Class to fetch data from a specified URL."""

    def __init__(self, url: str) -> None:
        self.url = url

    def fetch_data(self) -> str:
        """Fetch data from the provided URL."""
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        return None

    @staticmethod
    def convert_to_dataframe(data_text: Optional[str]) -> pd.DataFrame:
        """Convert data text to Pandas DataFrame."""
        if data_text:
            # Initialize a run
            neptune_run = get_neptune_run()
            df = pd.read_csv(StringIO(data_text))
            neptune_run["data/Training_data"].upload(File.as_html(df))
            return df
        return None


class DataIngestor:
    """Class responsible for data ingestion."""

    @staticmethod
    def ingest_data(url: str) -> pd.DataFrame:
        """Ingest data from the provided URL."""
        fetcher = DataFetcher(url)
        data_text = fetcher.fetch_data()
        return fetcher.convert_to_dataframe(data_text)


@step(experiment_tracker="neptune_experiment_tracker",enable_cache=False)
def ingest_data(url: str) -> pd.DataFrame:
    """ZenML Step: Ingests data from the provided URL"""
    return DataIngestor.ingest_data(url)


@step(enable_cache=False)
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

        # Initialize a run
        neptune_run = get_neptune_run()

        test_suite.save_html("Evidently_Reports/data_quality_suite.html")

        neptune_run["html/Data Quality Test"].upload("Evidently_Reports/data_quality_suite.html")
        
        alert_report(passed_tests, failed_tests, total_tests, "Data Quality Test", "Evidently_Reports/data_quality_suite.html", user_email)
    else:
        return curr_data