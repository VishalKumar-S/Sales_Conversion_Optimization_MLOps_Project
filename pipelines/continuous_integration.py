#continuous_integration.py
import pandas as pd
from steps.ingest_data import ingest_data, data_quality_validation
from steps.clean_data import clean_data, data_drift_validation
from steps.train_model import train_model, model_performance_validation
from zenml import pipeline, step
from steps.email_report import email_report
import logging
from steps.production_batch_data import production_batch_data
from steps.predict_prod_data import predict_prod_data

@pipeline(enable_cache=False)
def continuous_integration(path: str):
    #### FOR REFERENCE (ORIGINAL) DATASET ####

    ref_data = ingest_data("https://sale2.s3.us-east-2.amazonaws.com/KAG_conversion_data.csv")
    cleaned_data_ref, train_X, test_X = clean_data(ref_data)

    #### FOR CURRENT (PRODUCTION) DATASET ####

    batch_data = production_batch_data(path)
    data_quality_validated_data = data_quality_validation(batch_data)
    cleaned_data_curr, train_X, test_X = clean_data(data_quality_validated_data)
    cleaned_data_ref, cleaned_data_curr = data_drift_validation(cleaned_data_ref, cleaned_data_curr)
    predicted_ref_data, predicted_curr_data = predict_prod_data(cleaned_data_ref,cleaned_data_curr )
    model_performance_validation(predicted_ref_data, predicted_curr_data)



