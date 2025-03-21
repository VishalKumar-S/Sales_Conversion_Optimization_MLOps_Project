#ci_cd_pipeline.py
import pandas as pd
from steps.ingest_data import ingest_data
from steps.production_batch_data import data_quality_validation, data_stability_validation
from steps.clean_data import clean_data, data_drift_validation
from steps.train_model import train_model, model_performance_validation
from zenml import pipeline, step
from steps.alert_report import alert_report
import logging
from steps.production_batch_data import production_batch_data
from steps.predict_prod_data import predict_prod_data
import streamlit as st

@pipeline(enable_cache=True)
def ci_cd_pipeline(path: str, user_email="vishalkumar.s2022ai-ds@sece.ac.in"):

    #### FOR REFERENCE (TRAINING) DATASET ####
    ref_data = ingest_data("data/KAG_conversion_data.csv")
    cleaned_data_ref, train_X, test_X = clean_data(ref_data)

    #### FOR CURRENT (PRODUCTION) DATASET ####
    batch_data = production_batch_data(path)
    data_quality_validated_data = data_quality_validation(ref_data, batch_data, user_email)
    data_stability_validated_data = data_stability_validation(ref_data,data_quality_validated_data,user_email)
    cleaned_data_curr, train_X, test_X = clean_data(data_stability_validated_data)
    cleaned_data_ref, cleaned_data_curr = data_drift_validation(cleaned_data_ref, cleaned_data_curr, user_email)
    predicted_ref_data, predicted_curr_data = predict_prod_data(cleaned_data_ref,cleaned_data_curr)
    model_performance_validation(predicted_ref_data, predicted_curr_data, user_email)




    
