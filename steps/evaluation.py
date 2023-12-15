from zenml import step
import pandas as pd
import h2o
from typing import Dict

@step(enable_cache=False)
def evaluate(result: Dict[str, str]):
    # Unpack the result dictionary
    model_path = result['model_path']
    test_data_path = result['test_data_path']

    # Initialize H2O
    h2o.init()

    # Load the model from the MOJO file
    best_model = h2o.upload_mojo(model_path)

    # Load the test data from the CSV file
    test_data = pd.read_csv(test_data_path)

    # Convert Pandas DataFrame to H2O DataFrame
    h2o_test_data = h2o.H2OFrame(test_data)

    # Specify the response column
    response = "Approved_Conversion"

    # Predict on the test data
    predictions = best_model.predict(h2o_test_data)

    # Calculate evaluation metrics on the test data
    performance = best_model.model_performance(h2o_test_data)

    # Print the performance
    print(performance)
