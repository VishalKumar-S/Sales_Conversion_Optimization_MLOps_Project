import pandas as pd
from zenml import pipeline, step
import logging



@step(enable_cache = False)
def production_batch_data(data_path:str) -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        df = pd.read_csv(data_path)
        logging.info("Read csv file completed.")
        return df
    except Exception as e:
        logging.error(e)
        raise e