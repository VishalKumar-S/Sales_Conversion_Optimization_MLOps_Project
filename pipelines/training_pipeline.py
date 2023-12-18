#training_pipeline.py
import pandas as pd
from steps.ingest_data import ingest_data, data_quality_validation
from steps.clean_data import clean_data, data_drift_validation
from steps.train_model import train_model, model_performance_validation
from zenml import pipeline
from steps.email_report import email_report

@pipeline(enable_cache=False)
def train_pipeline(url: str):
    df = ingest_data(url)
    data_quality_validated_dataset = data_quality_validation(df)
    cleaned_data,reference_dataset, current_dataset = clean_data(df)
    reference_dataset, current_dataset = data_drift_validation(reference_dataset, current_dataset)
    train_df, test_df= train_model(cleaned_data)
    model_performance_validation(train_df, test_df)