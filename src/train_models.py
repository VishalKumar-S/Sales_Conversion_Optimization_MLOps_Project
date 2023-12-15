import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import joblib
import neptune
import datetime

# Generate a unique key for the model
timestamp = datetime.datetime.now().strftime("%d%H%M")
model_key = f"MOD{timestamp}"

model = neptune.init_model(
    name="Prediction model H20.ai AutoML",
    key= model_key,
    project="Vishal-Kumar-S/Sales-Conversion-Optimisation-MLOps-Project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5Mjg3OTM1OS01MjI2LTQzY2ItOTIzMi02YjIyMGMwZTUzMjkifQ==",
)

def train_h2o_automl(cleaned_dataset):
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

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    best_model = aml.leader
    print(lb)
    print(best_model)

    # Get model performance
    perf = best_model.model_performance()

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
