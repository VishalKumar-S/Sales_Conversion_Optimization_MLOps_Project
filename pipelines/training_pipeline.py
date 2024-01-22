from zenml import pipeline, step
from steps.ingest_data import ingest_data, data_quality_validation
from steps.clean_data import clean_data, data_drift_validation
from steps.train_model import train_model, model_performance_validation
from steps.alert_report import alert_report

class TrainPipelineBuilder:
    def __init__(self, url):
        self.url = url
        self.user_email="vishalkumar.s2022ai-ds@sece.ac.in"

    def build(self):
        df = ingest_data(self.url)
        data_quality_validated_dataset = data_quality_validation(df, self.user_email)
        cleaned_data, reference_dataset, current_dataset = clean_data(df)
        reference_dataset, current_dataset = data_drift_validation(reference_dataset, current_dataset, self.user_email)
        train_df, test_df = train_model(cleaned_data)
        model_performance_validation(train_df, test_df, self.user_email)

@pipeline(enable_cache=False)
def train_pipeline(url: str):
    builder = TrainPipelineBuilder(url)
    builder.build()
