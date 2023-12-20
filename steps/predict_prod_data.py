from zenml import step
from typing import Tuple, Annotated
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import matplotlib.pyplot as plt
import logging

from neptune.types import File
import neptune
from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run

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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ref_data['prediction'], ref_data['Approved_Conversion'], label='Reference Data')
        ax.scatter(curr_data['prediction'], curr_data['Approved_Conversion'], label='Current Data')
        ax.set_title('Predictions vs Actual')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Actual')
        ax.legend()
        fig.savefig('CML_Reports/predictions_scatter_plot.png')
        return fig

    @staticmethod
    def generate_residuals_plot(ref_data, curr_data):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ref_data['prediction'], ref_data['prediction'] - ref_data['Approved_Conversion'],label='Reference Data')
        ax.scatter(curr_data['prediction'], curr_data['prediction'] - curr_data['Approved_Conversion'], label='Current Data')
        ax.set_title('Residuals Plot')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.legend()
        fig.savefig('CML_Reports/residuals_plot.png')
        return fig


@step(experiment_tracker="neptune_experiment_tracker",enable_cache=False)
def predict_prod_data(ref_data: pd.DataFrame, curr_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        PredictionFacade.init_h2o()
        model_path = "models/best_model.zip/GLM_1_AutoML_1_20231214_194418.zip"
        trained_model = PredictionFacade.load_trained_model(model_path)

        ref_data['prediction'] = PredictionFacade.make_predictions(ref_data, trained_model)
        curr_data['prediction'] = PredictionFacade.make_predictions(curr_data, trained_model)

        # Initialize a run
        neptune_run = get_neptune_run()

        fig1 = PredictionFacade.generate_scatter_plot(ref_data, curr_data)
        neptune_run["visuals/scatter_plot"].upload(File.as_html(fig1))

        fig2 =PredictionFacade.generate_residuals_plot(ref_data, curr_data)
        neptune_run["visuals/residuals_plot"].upload(File.as_html(fig2))

        print(ref_data.head(5))
        print(curr_data.head(5))
            
        return ref_data, curr_data
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e
