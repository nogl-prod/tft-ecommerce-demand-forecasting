import yaml
import logging
from sqlalchemy import create_engine
from typing import Dict
import time
from pandas import DataFrame, concat, read_sql
from contextlib import contextmanager
# from one_drive_client import OneDriveAPI

class DatabaseManager:
    """
    A class to manage data in a PostgreSQL database.

    The DatabaseManager reads configuration data from a YAML file and provides methods to
    upload a pandas DataFrame to a PostgreSQL database and import data from the database.

    Example usage:

        # Initialize the DatabaseManager with the client_name
        manager = DatabaseManager(client_name="example_client")

        # Upload a DataFrame to the database
        manager.upload_dataframe(df, table_name="example_table", schema="public")

        # Import data from a table in the database
        imported_df = manager.import_data_from_aws_rds(table_name="example_table", schema="public")

        # Close the database connection
        manager.close_connection()

    Attributes:
        params (dict): A dictionary containing the configuration data for the database.
        engine (sqlalchemy.engine.Engine): A SQLAlchemy engine instance to manage the database connection.
    """
    def __init__(self, client_name: str, config_filename: str = '/opt/ml/processing/scripts/Inference/config/configAWSRDS.yaml'):
        self.params = self.config(client_name, config_filename)
        self.engine = create_engine(
            f"postgresql://{self.params.get('user')}:{self.params.get('password')}@{self.params.get('host')}:5432/{self.params.get('database')}",
            echo=False
        )

    def config(self, section: str, filename: str) -> Dict[str, str]:
        """
        Read configuration from a YAML file.

        Args:
            section (str): The section in the config file to read.
            filename (str): The name of the YAML config file.
        
        Returns:
            dict: Dictionary containing the configuration data.
        
        Raises:
            Exception: If the specified section is not found in the file.
        """
        with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)

        if section in config_data:
            return config_data[section]
        else:
            raise Exception(f'Section {section} not found in the {filename} file')

    def upload_data(self, df: DataFrame, table_name: str, schema: str, if_exists: str = 'replace', index: bool = False, chunksize: int = 1000, method: str = "multi") -> None:
        """
        Upload a pandas DataFrame to a PostgreSQL database.

        Args:
            df (DataFrame): The DataFrame to be uploaded.
            table_name (str): The name of the table in the database.
            schema (str): The schema in the database.
            if_exists (str, optional): The behavior if the table exists in the database. Options are 'fail', 'replace', and 'append'. Default is 'replace'.
            index (bool, optional): Whether to write the DataFrame index to the database. Default is False.
            chunksize (int, optional): Number of rows to be written at a time. Default is 1000.
            method (str, optional): Method to use when writing to the database. Default is "multi".
        """
        df.to_sql(table_name, con=self.engine, schema=schema, if_exists=if_exists, index=index, chunksize=chunksize, method=method)
        logging.info(f"Uploaded {table_name} to database")

    def close_connection(self) -> None:
        """
        Close the database connection.
        """
        self.engine.dispose()

    @contextmanager
    def timer(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            logging.info(f"{label}: {end - start} seconds")

    def import_data_from_aws_rds(self, table_name: str, schema: str, chunksize: int = 5000) -> DataFrame:
        """
        Import data from an AWS RDS PostgreSQL table to a Pandas DataFrame.

        Args:
            table (str): The name of the table in the database.
            schema (str): The schema in the database.
            chunksize (int, optional): Number of rows to be read at a time. Default is 5000.

        Returns:
            DataFrame: A DataFrame containing the imported data.
        """
        with self.timer(f"Importing {schema}.{table_name}"):
            chunks = []
            for chunk in read_sql(f"SELECT * FROM {schema}.{table_name}", con=self.engine, chunksize=chunksize):
                chunks.append(chunk)
            return concat(chunks)

# if __name__ == "__main__":
#     import pandas as pd
#     client_name = "stoertebekker"
#     onedrive_api = OneDriveAPI(client_name = "stoertebekker", config_filename = r'C:\Users\tuhin\Repo\Data-Processing-Deployment-Test\Inference\config\configOneDrive.yaml')
#     bundle_matrix = onedrive_api.download_file_by_relative_path("NOGL_shared/"+client_name+"/Bundel-Product-Matrix-"+client_name+".xlsx")
#     uploader = DatabaseManager(client_name = "stoertebekker", config_filename = r'C:\Users\tuhin\Repo\Data-Processing-Deployment-Test\Inference\config\configAWSRDS.yaml')
#     uploader.upload_data(bundle_matrix, schema="forecasts", table_name="bundle_matrix")
