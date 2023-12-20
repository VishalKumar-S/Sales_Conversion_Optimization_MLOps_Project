import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import joblib
import neptune
import datetime
import os

# Example usage:
timestamp = datetime.datetime.now().strftime("%d%H%M")
model_key = f"MOD{timestamp}"

class ModelFactory:
    @staticmethod
    def create_model(name, key, project, api_token):
        model = neptune.init_model(
            name=name,
            key=key,
            project=project,
            api_token=api_token,
        )
        return model

def train_h2o_automl(cleaned_dataset):

    model = ModelFactory.create_model(
    name="Prediction model H20.ai AutoML",
    key=model_key,
    project="Vishal-Kumar-S/Sales-Conversion-Optimisation-MLOps-Project",
    api_token=os.environ.get('API_TOKEN'),)

    # Initialize H2O
    h2o.init()
    # Load data
    data = cleaned_dataset
    # Convert Pandas DataFrame to H2O DataFrame
    h2o_data = h2o.H2OFrame(data)

    # Specify the response and predictor columns
    response = "Approved_Conversion"
    predictors = data.columns.drop(response).tolist()

    # Split the data into train and test sets for model validation
    train, test = h2o_data.split_frame(ratios=[0.8])

    # Initialize and train the H2OAutoML model
    aml = H2OAutoML(max_models=20, seed=42)
    aml.train(x=predictors, y=response, training_frame=train)

    # Save the best model
    model_path = "./models/best_model.zip"
    aml.leader.download_mojo(model_path, get_genmodel_jar=True)

    # Get model performance
    perf = aml.leader.model_performance()

    # Log model performance metrics to Neptune
    model["r2"].log(perf.r2())
    model["mse"].log(perf.mse())
    model["rmse"].log(perf.rmse())
    model["rmsle"].log(perf.rmsle())
    model["mae"].log(perf.mae())

    # Get predictions for train and test datasets
    train_pred = aml.predict(train)
    test_pred = aml.predict(test)

    # Convert H2OFrames to pandas DataFrames
    train_df = train.as_data_frame()
    test_df = test.as_data_frame()

    # Add predictions to the original train and test datasets
    train_df['prediction'] = train_pred.as_data_frame()
    test_df['prediction'] = test_pred.as_data_frame()

    return train_df, test_df



