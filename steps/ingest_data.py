import numpy as numpy
import pandas as pd
from zenml import step
import requests
from io import StringIO
import logging
from steps.email_report import email_report
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import DataQualityTestPreset
import logging
from steps.clean_data import clean_data

class FetchData:

    def __init__(self,url: str)->None:
        # Define the URL
        self.url =url

    def fetch_data(self):
        response = requests.get(self.url)
        # Check if the URL is accessible
        if response.status_code == 200:
            logging.info("URL is accessible. Downloading the data...")
            return response.text
        else:
            logging.info(f"Failed to access the URL. Status code: {response.status_code}")
            return None
        
    @staticmethod
    def get_data(data_text):
        try:
            if data_text:
                # Read the data into a pandas DataFrame
                df = pd.read_csv(StringIO(data_text))
                logging.info("Data ingested successfully")
                return df
            else:
                logging.error("No data received from the URL.")
                return None
        
        except Exception as e:
            logging.error(f"Error processing the data: {e}")
            return None

@step(enable_cache=False)
def ingest_data(url:str):
    try:
        fetch = FetchData(url)
        data_text = fetch.fetch_data()
        df=fetch.get_data(data_text)
        return df
    except Exception as e:
        logging.error(e)
        raise e
    

@step(enable_cache=False)
def data_quality_validation(curr_data: pd.DataFrame)->pd.DataFrame:
    test_suite = TestSuite(tests=[DataQualityTestPreset(),])
    test_suite.run(reference_data=None, current_data=curr_data)
    threshold = test_suite.as_dict()['summary']['success_tests']/test_suite.as_dict()['summary']['total_tests']
    test_name = "Data Quality Test"
    passed_tests = test_suite.as_dict()['summary']['success_tests']
    failed_tests =test_suite.as_dict()['summary']['failed_tests']
    total_tests= test_suite.as_dict()['summary']['total_tests']
    logging.info(f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, out of {total_tests} tests conducted.")
    if(threshold<0.85):
        test_suite.save_html("Reports/data_quality_suite.html")
        email_report(passed_tests, failed_tests, total_tests, "Data Quality Test", "Reports/data_quality_suite.html")
    else:
        logging.info(f"All Data Quality checks passed")
        return curr_data
        
        



