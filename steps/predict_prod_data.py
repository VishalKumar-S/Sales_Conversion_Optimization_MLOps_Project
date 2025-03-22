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
    """The PredictionFacade class follows facade patter, where we encapsulates all prediction-related logic, making it reusable and modular, and also the class is intended to abstract away complex logic and provide a simple interface for prediction-related tasks
    """
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
        """
        Ideal Scenario:
        If the model is perfect, all points would lie on a straight diagonal line (y = x), meaning the predicted values exactly match the actual values.

        Good Model:
        Points should be clustered closely around the diagonal line. The closer they are, the better the model’s predictions.

        Poor Model:
        If the points are scattered far from the diagonal line, the model’s predictions are not accurate.

        Comparison Between Reference and Current Data:

        If the current data points deviate significantly from the reference data points in the plot, it could indicate:

        1) Data drift (the new data is different from the training data).

        2) Model performance degradation over time.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ref_data['prediction'], ref_data['Approved_Conversion'], label='Reference Data')
        ax.scatter(curr_data['prediction'], curr_data['Approved_Conversion'], label='Current Data')
        ax.set_title('Predictions vs Actual')
        ax.set_title('Scatter Plot')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Actual')
        ax.legend()
        fig.savefig('CML_Reports/predictions_scatter_plot.png')
        return fig

    @staticmethod
    def generate_residuals_plot(ref_data, curr_data):
        """
        Ideal Scenario:
        Residuals should be randomly scattered around zero with no clear pattern. This indicates that the model’s errors are random and not systematic.

        Patterns in Residuals:
        If you see a clear pattern (e.g., a curve or trend), it suggests that the model is biased and not capturing some relationship in the data.

        For example:
        A U-shaped pattern might indicate that the model is underfitting.

        A funnel shape (residuals spread out as predictions increase) might indicate heteroscedasticity (non-constant variance).

        Outliers:
        Points that are far from zero indicate large prediction errors. These could be outliers or instances where the model performs poorly.

        Comparison Between Reference and Current Data:
        
        If the residuals for the current data are significantly larger or show a different pattern compared to the reference data, it could indicate:

        1) Model performance degradation.

        2) Changes in the data distribution (data drift).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ref_data['prediction'], ref_data['prediction'] - ref_data['Approved_Conversion'],label='Reference Data')
        ax.scatter(curr_data['prediction'], curr_data['prediction'] - curr_data['Approved_Conversion'], label='Current Data')
        ax.set_title('Residuals Plot')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.legend()
        fig.savefig('CML_Reports/residuals_plot.png')
        return fig


@step(experiment_tracker="neptune_experiment_tracker",enable_cache=True)
def predict_prod_data(ref_data: pd.DataFrame, curr_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        PredictionFacade.init_h2o()
        model_path = "models/best_model.zip/StackedEnsemble_BestOfFamily_1_AutoML_1_20240122_204822.zip"
        trained_model = PredictionFacade.load_trained_model(model_path)

        ref_data['prediction'] = PredictionFacade.make_predictions(ref_data, trained_model)
        curr_data['prediction'] = PredictionFacade.make_predictions(curr_data, trained_model)

        neptune_run = get_neptune_run()

        fig1 = PredictionFacade.generate_scatter_plot(ref_data, curr_data)
        neptune_run["visuals/scatter_plot"].upload(File.as_html(fig1))

        fig2 =PredictionFacade.generate_residuals_plot(ref_data, curr_data)
        neptune_run["visuals/residuals_plot"].upload(File.as_html(fig2))

            
        return ref_data, curr_data
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e
