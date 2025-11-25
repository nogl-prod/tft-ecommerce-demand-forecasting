import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Union
import boto3
import io

# Set up logging to output to console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)
class DataProcessor:

    
    def __init__(self, db_conn_str: Optional[str] = None, file_path: Optional[str] = None, log_to_sql: bool = False, use_s3: bool = True) -> None:
        """
        Initialize a new instance of the DataProcessor class.

        Args:
            db_conn_str (str): The connection string for the SQL database.
            log_to_sql (bool): Flag to control whether to log processing steps to the SQL database. Default is True.
        """

        self.log_to_sql = log_to_sql
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(log_formatter)
        self.logger = logging.getLogger()
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)

        if db_conn_str is not None:
            self.engine = create_engine(db_conn_str)
        
        if file_path is None:
            self.file_path = "/opt/ml/processing/input2/"
        else:
            self.file_path = file_path
        self.use_s3 = use_s3
        self.use_filesystem = not use_s3

    def _read_data_from_filesystem(self, filename: str) -> pd.DataFrame:
        """
        Reads data from a CSV file located on the filesystem.

        Args:
            filename (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Data read from the CSV file as a Pandas DataFrame.
        """
        return pd.read_csv(filename)

    def _read_data_from_postgres(self, query: str) -> pd.DataFrame:
        """
        Reads data from a PostgreSQL database using a SQL query and returns the results as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the results of the SQL query.
        """
        # Use pandas to execute the SQL query and read the results into a DataFrame
        result_df = pd.read_sql_query(query, con=self.engine)

        # Return the DataFrame containing the results
        return result_df

    def _read_data_from_s3(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file from an S3 bucket and returns its contents as a pandas DataFrame.

        Args:
            filename (str): The S3 path to the input CSV file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the input CSV file.
        """
        # Create an S3 client
        s3 = boto3.client("s3")

        # Extract the bucket name and key from the input filename
        bucket_name, key = self._parse_s3_path(file_path)

        # Print some diagnostic messages
        logging.info('Requesting S3 data')
        logging.info(file_path)

        # Download the CSV file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)

        # Print some diagnostic messages
        logging.info('Loading S3 Metadata')
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        logging.info('Saving data to df')

        # If the download was successful, read the CSV content into a pandas DataFrame
        if status == 200:
            # Print a diagnostic message indicating that the S3 get_object request was successful
            logging.info(f"Successful S3 get_object response. Status - {status}")

            # Read the contents of the CSV file into a string variable
            csv_file = response["Body"].read().decode("utf-8")

            # Create a StringIO object from the CSV string
            csv_content = io.StringIO(csv_file)
        else:
            # If the S3 get_object request was unsuccessful, print a diagnostic message
            logging.info(f"Unsuccessful S3 get_object response. Status - {status}")

            # Set csv_content to None so that it can be used as a flag to indicate an error condition
            csv_content = None

        # Return the DataFrame
        return pd.read_csv(csv_content)

    def _parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parses an S3 path string and returns the bucket name and key as a tuple.

        Args:
            s3_path (str): The S3 path string to parse.

        Returns:
            tuple: A tuple containing the bucket name and key parsed from the S3 path string.
        """
        # Check that the input string starts with "s3://"
        if not s3_path.startswith("s3://"):
            raise ValueError("Invalid S3 path")

        # Split the path string into two parts: the bucket name and the key
        path_parts = s3_path[5:].split("/", 1)
        if len(path_parts) != 2:
            raise ValueError("Invalid S3 path")

        # Return the bucket name and key as a tuple
        return path_parts[0], path_parts[1]

    def import_data(self, date_str: Optional[str] = None, file_type: Optional[str] = None, s3: Optional[str] = None, filesystem: Optional[str] = None, query: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Read data from a CSV file for the given date and file type or keyword, preprocess it, and optionally write it to a SQL database.

        Args:
            date_str (Optional[str]): The current date in the format YYYY-MM-DD. Default is None.
            file_type (Optional[str]): The type of file to process. Options: "TopSeller", "LongTail", "Kicked". Default is None.
            keyword (Optional[str]): A custom keyword to use as the filename. Default is None.

        Returns:
            (Optional[pd.DataFrame]) The preprocessed data. Returns None if an error occurs.

        Raises:
            ValueError: If neither date_str nor keyword are provided or if date_str is provided but file_type is not.
        """
        if not s3 and not date_str and not filesystem:
            raise ValueError("Either date_str or s3 must be provided")

        if date_str and not file_type:
            raise ValueError("file_type must be provided if date_str is provided")

        if date_str and not self._validate_date(date_str, date_format="%Y-%m-%d"):
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")

        if file_type and file_type not in ["TopSeller", "LongTail", "Kicked"]:
            raise ValueError("Invalid file type. Options: 'TopSeller', 'LongTail', 'Kicked'")

        file_path = self._construct_filename(date_str, file_type, s3, filesystem)

        if self.log_to_sql:
            log_file_type = file_type if file_type else "unknown"
            self._log_processing_step(f"read {log_file_type} csv", file_path)

        try:
            if self.use_s3:
                self.data = self._read_data_from_s3(file_path = file_path)
            elif self.use_filesystem:
                self.data = self._read_data_from_filesystem(file_path)
            else:
                self.data = self._read_data_from_postgres(file_path)
        except Exception as e:
            self.logger.exception(f"Error reading data from {file_path}")
            raise ValueError(f"Error reading data from {file_path}: {str(e)}")

        self._preprocess_data()

        logging.info(f"Successfully preprocessed data")

        return self.data


    def _construct_filename(self, date_str: Optional[str] = None, file_type: Optional[str] = None, s3: Optional[str] = None, filesystem: Optional[str] = None) -> str:
        """
        Construct a filename based on the provided date_str, file_type, and keyword.
        
        Args:
            date_str (Optional[str]): The current date in the format YYYY-MM-DD. Default is None.
            file_type (Optional[str]): The type of file to process. Options: "TopSeller", "LongTail", "Kicked". Default is None.
            keyword (Optional[str]): A custom keyword to use as the filename. Default is None.
            
        Returns:
            str: The constructed filename.
            
        If both date_str and keyword are provided, the keyword takes precedence.
        If neither date_str nor keyword are provided, a ValueError is raised.
        If date_str is provided but file_type is not, a ValueError is raised.
        """
        if s3:
            filename = f"{s3}"
        elif filesystem:
            filename = f"{filesystem}"
        elif date_str and file_type:
            filename = f"{self.file_path}{date_str.replace('-', '')}_{file_type}_consolidated_cutoff.csv"
        elif date_str and not file_type:
            raise ValueError("file_type must be provided if date_str is provided")
        else:
            raise ValueError("Either date_str or keyword must be provided")

        return filename


    def _validate_date(self, date_str: str, date_format: str = "%Y-%m-%d") -> bool:
        """
        Validates a date string against a specified date format.

        Args:
            date_str (str): The date string to validate.
            date_format (str): The date format string to use for validation. Default is "%Y-%m-%d".

        Returns:
            bool: True if the date string is valid and matches the specified format, False otherwise.
        """
        try:
            # Use the datetime.strptime() method to try to parse the date string using the specified format
            datetime.strptime(date_str, date_format)

            # If the parse is successful, return True
            return True
        except ValueError:
            # If the parse fails, return False
            return False

    def _preprocess_data(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if "Unnamed: 0" not in self.data.columns:
            raise ValueError("Input data must contain 'Unnamed: 0' column")
        self.data.drop(columns="Unnamed: 0", inplace=True)

    def _log_processing_step(self, step: str, path: str) -> None:
        upload_sql = pd.DataFrame([[step, path]], columns=["step", "path"])
        upload_sql.to_sql("read_data1", con=self.engine, schema="forecast_test", if_exists="replace", index=False, chunksize=1000, method="multi")

def main() -> None:
    processor = DataProcessor(use_s3= False)
    processed_data = processor.process_data( filesystem = "preprocessed_stoertebekker_modified.csv")
 

if __name__ == "__main__":
    main()
