from zenml import step
import pandas as pd
import h2o
from typing import Dict

class ModelFactory:
    @staticmethod
    def load_model(model_path):
        h2o.init()  # Initialize H2O
        return h2o.upload_mojo(model_path)

    @staticmethod
    def load_test_data(test_data_path):
        return pd.read_csv(test_data_path)

    @staticmethod
    def convert_to_h2o(data):
        return h2o.H2OFrame(data)

@step(enable_cache=False)
def evaluate(result: Dict[str, str]):
    # Unpack the result dictionary
    model_path = result['model_path']
    test_data_path = result['test_data_path']

    # Create model instance
    best_model = ModelFactory.load_model(model_path)

    # Load test data
    test_data = ModelFactory.load_test_data(test_data_path)

    # Convert test data to H2O Frame
    h2o_test_data = ModelFactory.convert_to_h2o(test_data)

    # Specify the response column
    response = "Approved_Conversion"

    # Predict on the test data
    predictions = best_model.predict(h2o_test_data)

    # Calculate evaluation metrics on the test data
    performance = best_model.model_performance(h2o_test_data)

    # Print the performance
    print(performance)
