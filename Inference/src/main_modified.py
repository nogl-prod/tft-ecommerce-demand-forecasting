# Import necessary libraries and modules
import argparse
import logging
import traceback  
from contextlib import contextmanager
import time
from typing import Optional
from utils import ModelForecast, DataWrangling, TFTModelHandler, DataProcessor, DatabaseManager, OneDriveAPI, SalesDataUnbundler, ForecastDataUnbundler
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
# Set up logging to output to console
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)
# Argument parser for command-line options
# S3_BUCKET_PATH = ["s3://tft-training-data/stoertebekker/20230605_TopSeller_consolidated_cutoff.csv", "s3://tft-training-data/stoertebekker/20230605_LongTail_consolidated_cutoff.csv","s3://tft-training-data/stoertebekker/20230605_Kicked_consolidated_cutoff.csv"]
S3_BUCKET_PATH = ["s3://fcb-bucket/dataset/20230607_TopSeller_consolidated_cutoff.csv"]
LAST_OBSERVATION_DAY = "2023-06-04"
# MODEL_PATH = ["/opt/ml/processing/models/trained_models/tft/no-rolling/stoertebekker/","/opt/ml/processing/models/trained_models/tft/rolling/stoertebekker/","no_model"]      # Please dont change the path as it is hard coded to check no-rolling
MODEL_PATH = ["/opt/ml/processing/models/models/tft/no-rolling/"]      # Please dont change the path as it is hard coded to check no-rolling
CLIENT_NAME = "fcb"
MAX_ENCODER_LENGTH = 380
MAX_PREDICTION_LENGTH = 90
TABLE_NAME = "forecast"
BUNDLE_MATRIX = "bundle_matrix"
BUNDLE = "False"
UPLOAD_PLOTS = False
SCHEMA_FORECAST = "forecasts"
MODEL_DATE = "20230608"
SCHEMA_HISTORIC = "transformed"
SCHEMA_XAI = "xai"

parser = argparse.ArgumentParser(description="This script processes, analyzes, and forecasts data using a specified model.")

# New arguments for your datasets and models
parser.add_argument(
    "--datasets", 
    nargs='+',
    default=S3_BUCKET_PATH, 
    help="A list of paths to the datasets you want to process."
)
parser.add_argument(
    "--models", 
    nargs='+',
    default=MODEL_PATH, 
    help="A list of paths to the models you want to use for forecasting."
)

parser.add_argument(
    "--last_observation_day",  
    type=str, 
    default=LAST_OBSERVATION_DAY, 
    help="The last day of observation in the time series data. This is used to determine the time period for the forecasting."
)

parser.add_argument(
    "--client_name", 
    type=str, 
    default=CLIENT_NAME, 
    help="The name of the client. This is used for identification purposes in the database."
)

parser.add_argument(
    "--upload_plots", 
    type=bool, 
    default=UPLOAD_PLOTS, 
    help="A boolean flag indicating whether to upload the forecast plots to the specified S3 bucket."
)

parser.add_argument(
    "--max_encoder_length", 
    type=int, 
    default=MAX_ENCODER_LENGTH, 
    help="The maximum length of the encoder in the forecasting model. This sets an upper limit on the number of input data points for the model."
)

parser.add_argument(
    "--max_prediction_length", 
    type=int, 
    default=MAX_PREDICTION_LENGTH, 
    help="The maximum length of the forecast period. This sets an upper limit on the number of days for which the model will produce forecasts."
)

parser.add_argument(
    "--s3_bucket", 
    type=str, 
    default=None, 
    help="The name of the S3 bucket to which the forecast plots will be uploaded."
)

parser.add_argument(
    "--s3_folder_path", 
    type=str, 
    default=None, 
    help="The folder path in the S3 bucket where the forecast plots will be stored."
)

parser.add_argument(
    "--table_name", 
    type=str, 
    default=TABLE_NAME, 
    help="The name of the database table where the forecast results will be stored."
)

parser.add_argument(
    "--bundle", 
    type=str, 
    default=BUNDLE, 
    help="The name of the database table where the forecast results will be stored."
)

parser.add_argument(
    "--date_str", 
    type=str, 
    default=MODEL_DATE, 
    help="The date for which the model should be loaded."
)

args = parser.parse_args()

# Create a context manager for timing sections of your code
@contextmanager
def log_time(section_name: str):
    """
    Context manager for timing sections of your code. 
    It logs completion time on successful completion of the with block.
    """
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"{section_name} completed in {elapsed_time} seconds")

def process_dataset_model(data_model_date):
    dataset, model_path, date_str = data_model_date
    uploader = DatabaseManager(args.client_name)
    try:
        with log_time("Importing Data"):    
            # Importing Data
            processor = DataProcessor()
            processed_data = processor.import_data(s3 = dataset)

        # Data wrangling
        with log_time("Data wrangling"):
            # Data wrangling
            dw = DataWrangling(processed_data)
            preprocessed_data, last_day, training_cutoff, end_of_timeseries = dw.prepare_data_for_inference(args.last_observation_day)

        # Model handling
        # TODO: Change load_specific_checkpoint to best_tft_checkpoint for production
        with log_time("Model handling"):
            if model_path != "no_model":
                logging.info(f"Searching for  {model_path} ")
                model_handler = TFTModelHandler(model_path, metric="val_loss")
                model, model_type = model_handler.load_specific_checkpoint_from_s3(model_path=model_path)
            else:
                logging.info(f"Assigning model_type = extreme_long_tail ")
                model_type = "extreme_long_tail"

        # Model forecasting
        with log_time(f"Starting {model_type} model forecasting"):
            if model_type == "extreme_long_tail":
                forecasting = ModelForecast()
                results = forecasting.get_extreme_long_tail_forecasts(data = processed_data, last_day= last_day, end_of_timeseries=end_of_timeseries, model_type=model_type)
            else:
                forecasting = ModelForecast(model=model, model_type=model_type)
                results = forecasting.get_forecasts(
                    preprocessed_data,
                    first_prediction_time_idx = last_day+1,
                    time_idx="TVKR_time_idx",
                    max_encoder_length = args.max_encoder_length,
                    max_prediction_length = args.max_prediction_length
                )

#         # Data interpretation
#         if model_type != "extreme_long_tail":
#             with log_time("Data interpretation"):
#                 dfs_dict = forecasting.generate_interpretation(s3_bucket = args.s3_bucket, s3_folder_path= args.s3_folder_path, upload_plots=args.upload_plots)
#                 for table_name, df in dfs_dict.items():
#                     uploader.upload_data(df, schema=SCHEMA_XAI, table_name=table_name, if_exists="append")
#                 logging.info(f"Data interpretation of {model_type} completed")
            
        return results
        
    except Exception as e:
        logging.error(f"Error processing dataset {dataset} with model {model_path}: {e}")
        traceback.print_exc()
        return None

# Main function
def main(datasets: str, models: str, client_name: str, table_name: Optional[str] = None, bundle: Optional[bool] = True, date_str: Optional[str] = None) -> None:

    """
    This function is the main entry point of the program. It takes in four arguments:
    s3 (str): The S3 bucket path where the preprocessed data is stored.
    last_observation_day (str): The last day of the time series data.
    model_path (str): The path to the trained model checkpoint.
    client_name (str): The name of the client.

    The function performs the following steps:
    1. Data processing: The preprocessed data is loaded and processed.
    2. Data wrangling: The processed data is wrangled to prepare it for inference.
    3. Model handling: The trained model is loaded.
    4. Model forecasting: The model is used to generate forecasts.
    5. Data uploading: The forecast results are uploaded to a database.
    6. Data interpretation: The forecast results are interpreted and uploaded to a database.
    7. Connection closed: The database connection is closed.

    If an error occurs during any of the above steps, an error message is logged and the program is terminated.

    Args:
    s3 (str): The S3 bucket path where the preprocessed data is stored.
    last_observation_day (str): The last day of the time series data.
    model_path (str): The path to the trained model checkpoint.
    client_name (str): The name of the client.

    Returns:
    None
    """
    results_list = []
    with Pool(cpu_count()) as p:
        data_model_date_tuples = list(zip(datasets, models, [date_str]*len(datasets)))
        results_list = p.map(process_dataset_model, data_model_date_tuples)
    
    # Check for task failures
    if any(result.empty for result in results_list):
        logging.error("One or more tasks have failed. Aborting concatenation.")
        return
    
    # concatenate all results
    results = pd.concat(results_list)
    logging.info("All results concatenated")
    logging.info(bundle)
    # bundle = False
    try:
        if bundle == "True":
            # Importing plan data
            with log_time("Importing plan data"):
                onedrive_api = OneDriveAPI(client_name)
                bundle_matrix = onedrive_api.download_file_by_relative_path("NOGL_shared/"+client_name+"/Bundel-Product-Matrix-"+client_name+"_testing.xlsx")
            # Data uploading
            with log_time("Importing Shopify sales data"):
                downloader = DatabaseManager(client_name = "stoertebekker")
                shopify_sales_data = downloader.import_data_from_aws_rds(schema=SCHEMA_HISTORIC, table_name="shopify_sales")

            with log_time("Preparing Forecast bundle results"):
                target_columns = ['forecast_values']
                # Instantiate SalesDataUnbundler
                unbundler = ForecastDataUnbundler(bundle_matrix, target_columns)
                results = results.reset_index(drop=True)
                results = unbundler.unbundle_sales_data(df=results )
                results['forecast_values_bundle'] = results['forecast_values_bundle'].astype(np.float32)
                results['forecast_values_total'] = results['forecast_values_total'].astype(np.float32)
            
            with log_time("Preparing Historic bundle results"):
                shopify_sales_data['variant_id'] = shopify_sales_data['variant_id'].astype(str)
                bundle_matrix['variant_id'] = bundle_matrix['variant_id'].astype(str)
                # Create unbundler for historic results
                historic_unbundler = SalesDataUnbundler(bundle_matrix=bundle_matrix, 
                                        df_column_id='variant_id', 
                                        matrix_column_id='variant_id', 
                                        target_to_unbundle='lineitems_quantity', 
                                        date_column='daydate')
                historic_bundle_results = historic_unbundler.unbundle(df=shopify_sales_data)
        # Data uploading
        with log_time("Data uploading"):
            uploader = DatabaseManager(client_name)
            uploader.upload_data(results, schema=SCHEMA_FORECAST, table_name="forecast_best_model")
            if bundle == "True":
                uploader.upload_data(bundle_matrix, schema=SCHEMA_FORECAST, table_name=BUNDLE_MATRIX)
                uploader.upload_data(historic_bundle_results, schema=SCHEMA_HISTORIC, table_name="historic_bundle_results")
            logging.info("Data uploading completed")     
        # Close connection
        uploader.close_connection()
        logging.info("Connection closed")     


    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        traceback.print_exc()
        raise

# Invoke main function
if __name__ == "__main__":
    main(args.datasets, args.models, args.client_name, args.table_name, args.bundle, args.date_str)
