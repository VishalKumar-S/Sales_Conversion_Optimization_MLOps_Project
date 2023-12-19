from zenml import step
from typing import Tuple, Annotated
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import matplotlib.pyplot as plt
import logging

class PredictionFacade:
    @staticmethod
    def init_h2o():
        h2o.init()

    @staticmethod
    def load_trained_model(model_path):
        return h2o.import_mojo(model_path)

    @staticmethod
    def make_predictions(data, trained_model):
        return trained_model.predict(h2o.H2OFrame(data)).as_data_frame()

    @staticmethod
    def generate_scatter_plot(ref_data, curr_data):
        plt.figure(figsize=(10, 6))
        plt.scatter(ref_data['prediction'], ref_data['Approved_Conversion'], label='Reference Data')
        plt.scatter(curr_data['prediction'], curr_data['Approved_Conversion'], label='Current Data')
        plt.title('Predictions vs Actual')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')
        plt.legend()
        plt.savefig('CML_Reports/predictions_scatter_plot.png')

    @staticmethod
    def generate_residuals_plot(ref_data, curr_data):
        plt.figure(figsize=(10, 6))
        plt.scatter(ref_data['prediction'], ref_data['prediction'] - ref_data['Approved_Conversion'],
                    label='Reference Data')
        plt.scatter(curr_data['prediction'], curr_data['prediction'] - curr_data['Approved_Conversion'],
                    label='Current Data')
        plt.title('Residuals Plot')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.legend()
        plt.savefig('CML_Reports/residuals_plot.png')


@step(enable_cache=False)
def predict_prod_data(ref_data: pd.DataFrame, curr_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        PredictionFacade.init_h2o()
        model_path = "models/best_model.zip/GLM_1_AutoML_1_20231214_194418.zip"
        trained_model = PredictionFacade.load_trained_model(model_path)

        ref_data['prediction'] = PredictionFacade.make_predictions(ref_data, trained_model)
        curr_data['prediction'] = PredictionFacade.make_predictions(curr_data, trained_model)
        print(ref_data.head(5))
        print(curr_data.head(5))

        PredictionFacade.generate_scatter_plot(ref_data, curr_data)
        PredictionFacade.generate_residuals_plot(ref_data, curr_data)

        print(ref_data.head(5))
        print(curr_data.head(5))
        return ref_data, curr_data
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e
