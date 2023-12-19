from abc import ABC, abstractmethod
import pandas as pd
from zenml import step
import logging


class DataReader(ABC):
    @abstractmethod
    def read_data(self, data_path: str) -> pd.DataFrame:
        pass


class CSVDataReader(DataReader):
    def read_data(self, data_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(data_path)
            logging.info("Read CSV file completed.")
            return df
        except Exception as e:
            logging.error(e)
            raise e


class JSONDataReader(DataReader):
    def read_data(self, data_path: str) -> pd.DataFrame:
        try:
            # Logic to read JSON files
            # df = pd.read_json(data_path)
            logging.info("Read JSON file completed.")
            # return df
        except Exception as e:
            logging.error(e)
            raise e


class DataReaderFactory:
    def create_data_reader(self, data_format: str) -> DataReader:
        if data_format.lower() == 'csv':
            return CSVDataReader()
        elif data_format.lower() == 'json':
            return JSONDataReader()
        else:
            raise ValueError(f"Unsupported data format: {data_format}")


@step(enable_cache=False)
def production_batch_data(data_path: str, data_format: str = 'csv') -> pd.DataFrame:
    """
    Args:
        data_path (str): Path to the data file.
        data_format (str): Format of the data file (default: 'csv').
    Returns:
        df (pd.DataFrame): Processed DataFrame.
    """
    try:
        factory = DataReaderFactory()
        reader = factory.create_data_reader(data_format)
        return reader.read_data(data_path)
    except ValueError as ve:
        logging.error(ve)
        raise ve
    except Exception as e:
        logging.error(e)
        raise e
