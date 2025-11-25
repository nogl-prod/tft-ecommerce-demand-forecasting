import pandas as pd
from typing import Optional, Dict
import boto3
import os
import logging
import traceback  
import tempfile
import matplotlib.pyplot as plt
import torch
import numpy as np
from botocore.exceptions import BotoCoreError, NoCredentialsError


# Set up logging to output to console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

class ModelForecast:
    """
    A class to generate forecasts using a Temporal Fusion Transformer model.

    Attributes:
    - model: Temporal Fusion Transformer model to be used for forecasting.
    - model_type: str, type of the model, default is "top_seller".
    - data: DataFrame, stores the data used for prediction.
    - column_names: list, stores the column names of the data.
    - results: DataFrame, stores the results of the forecast.
    - predictions: list, stores the predictions from the model.
    - x: DataFrame, stores the input data for prediction.
    - index: DataFrame, stores the index of the prediction data.
    - decoder_lengths: list, stores the decoder lengths.
    - new_prediction_data: DataFrame, stores the new prediction data.

    Methods:
    - __init__: Initializes the Forecasting class with the given model.
    - build_results_dataframe: Prepares the DataFrame for storing the results.
    - get_forecasts: Generates forecast results using the provided data and model.
    - save_plots_to_s3: Saves the generated plots to the specified S3 path.
    - save_prediction_plots_to_s3: Saves the generated prediction plots to the specified S3 path.

    Usage:

        forecasting = ModelForecast(tft_model)
        results = forecasting.get_forecasts(
            data,
            first_prediction_time_idx,
            time_idx,
            max_encoder_length,
            max_prediction_length
        )
    """

    def __init__(self, model: object = None, model_type: str= "top_seller") -> None:
        """
        Initializes the Forecasting class with the given model.

        Args:
        - model: Temporal Fusion Transformer model to be used for forecasting.
        - model_type: str, type of the model, default is "top_seller".

        Returns:
            None

        Raises:
            ValueError: If model is not a model object.
        
        Usage:
        ```
        forecasting = ModelForecast(tft_model)
        ```
        """
        if not isinstance(model, object):
            raise ValueError("model must be a model object.")
        self.model = model
        self.model_type = model_type
        self.data = pd.DataFrame()
        self.column_names = []

    def set_dataframe_datatypes(self, results=None) -> pd.DataFrame:
        """
        Set the datatypes of the columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame to modify.

        Returns:
        pd.DataFrame: DataFrame with modified column datatypes.
        """
        if results is not None and not results.empty:
            self.results = results

        self.results['SC_variant_id'] = self.results['SC_variant_id'].astype(str)
        self.results['forecast_values'] = self.results['forecast_values'].astype(np.float32)
        self.results['quantile'] = self.results['quantile'].astype(np.int32)
        # self.results['SC_product_category_number'] = self.results['SC_product_category_number'].astype(np.int32)
        self.results['SC_product_category'] = self.results['SC_product_category'].astype(str)
        #self.results['SC_product_id'] = self.results['SC_product_id'].astype(np.int64)
        # self.results['TVUR_shopify_lineitems_quantity'] = self.results['TVUR_shopify_lineitems_quantity'].astype(np.int32)
        #self.results['SC_variant_sku'] = self.results['SC_variant_sku'].astype(str)
        self.results['TVKC_daydate'] = pd.to_datetime(self.results['TVKC_daydate'])
        self.results['TVKR_time_idx'] = self.results['TVKR_time_idx'].astype(np.int32)
        self.results['model_date'] = pd.to_datetime(self.results['model_date'])
        self.results['timesteps'] = self.results['timesteps'].astype(np.int32)
        self.results['SC_model_type'] = self.results['SC_model_type'].astype(str)
        if results is not None and not results.empty:
            return self.results

    def build_results_dataframe(self, first_prediction_time_idx, max_prediction_length, time_idx, predictions = None) -> None:
        """
        Build the results DataFrame based on the provided input data.
        """
        if predictions:
            self.predictions = predictions
        
        results = pd.DataFrame(columns=['SC_variant_id', 
                                        '0',
                                        '1',
                                        '2',
                                        '3',
                                        '4',
                                        '5',
                                        '6'])

        # load data in new format
        for i in self.index.index:
            p = pd.DataFrame(self.predictions[i].numpy())
            p["SC_variant_id"] = self.index.iloc[i,:].SC_variant_id
            p.rename(columns={0:"0",
                            1:"1",
                            2:"2",
                            3:"3",
                            4:"4",
                            5:"5",
                            6:"6"}, inplace=True)
            p = p[['SC_variant_id',
                '0', 
                '1', 
                '2', 
                '3', 
                '4',
                '5', 
                '6']]
            
            
            results = pd.concat([results, p], ignore_index=True)
            
        # finish building time index
        time_idx_column = []
        for i in range(len(self.x.get("decoder_time_idx"))):
            time_idx_column = time_idx_column + list(range(first_prediction_time_idx, first_prediction_time_idx+max_prediction_length,1))
        results[time_idx] = time_idx_column
        # columns_to_add = ['SC_product_category_number', 'SC_product_category', 'TVKC_daydate','TVKR_time_idx', 'SC_variant_id']
        columns_to_add = [ 'SC_product_category', 'TVKC_daydate','TVKR_time_idx', 'SC_variant_id']
        renamed_columns_df = self.new_prediction_data[columns_to_add]
        results = results.merge(renamed_columns_df, on=['TVKR_time_idx', 'SC_variant_id'], how='left')
        # Add the model_date column
        self.first_date =  results['TVKC_daydate'][0]
        results['model_date'] = self.first_date
        first_timestep = results['TVKR_time_idx'].iloc[0]
        results['timesteps'] = results['TVKR_time_idx'].apply(lambda x: x - first_timestep + 1)
        # Add the SC_model_type column with the value from self.model_type
        results['SC_model_type'] = self.model_type

        # Melt the DataFrame
#         results = results.melt(id_vars=['SC_model_type', 'TVKR_time_idx', 
#                              'SC_product_category_number', 'SC_product_category', 'SC_variant_id', 
#                              'TVKC_daydate', 'model_date', 'timesteps'],
#                             value_vars=[str(i) for i in range(7)], 
#                             var_name='quantile', 
#                             value_name='forecast_values')
        results = results.melt(id_vars=['SC_model_type', 'TVKR_time_idx', 
                             'SC_product_category', 'SC_variant_id', 
                             'TVKC_daydate', 'model_date', 'timesteps'],
                            value_vars=[str(i) for i in range(7)], 
                            var_name='quantile', 
                            value_name='forecast_values')
        results.sort_values(by=['TVKR_time_idx', 'quantile'], inplace=True)
        logging.info("Number of unique time series in encoder_data after prediction: %s", len(results["SC_variant_id"].unique()))
        logging.info("Number of unique time series in decoder_data after prediction: %s", len(results["SC_variant_id"].unique()))

        # rearrange column sorting
        cols = results.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        results = results[cols]
            
        self.results= results
        self.set_dataframe_datatypes()
    
    def get_forecasts(self, data: pd.DataFrame, first_prediction_time_idx: int, time_idx: str, max_encoder_length: int, max_prediction_length: int) -> pd.DataFrame:
        training_cutoff = first_prediction_time_idx - 1

        encoder_data = data[lambda x: (training_cutoff >= x[time_idx]) & (x[time_idx] > (training_cutoff - max_encoder_length))]
        decoder_data = data[lambda x: (x[time_idx] > (training_cutoff)) & (x[time_idx] < (first_prediction_time_idx + max_prediction_length))]
        self.new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        # Print the number of unique time series in the encoder_data and decoder_data
        logging.info("Number of unique time series in encoder_data: %s", len(encoder_data["SC_variant_id"].unique()))
        logging.info("Number of unique time series in decoder_data: %s", len(decoder_data["SC_variant_id"].unique()))


        self.predictions, self.x, self.index, self.decoder_lengths = self.model.predict(
                                                                                                                self.new_prediction_data,
                                                                                                                mode="quantiles",
                                                                                                                return_x=True,
                                                                                                                return_index=True,
                                                                                                                return_decoder_lengths=True
                                                                                                            )
    
        

        self.build_results_dataframe(first_prediction_time_idx, max_prediction_length, time_idx)
        
        return self.results
    
    def get_extreme_long_tail_forecasts(self, data: pd.DataFrame, last_day: int, end_of_timeseries: int, model_type: str) -> pd.DataFrame:
        forecasts_extreme_long_tail = data[(data["TVKR_time_idx"] > last_day) & 
                                                            (data["TVKR_time_idx"] <= end_of_timeseries)][["TVKR_time_idx", "SC_variant_id", 'SC_product_category', 'TVKC_daydate']].copy()
        to_merge = data[data["TVKR_time_idx"] == last_day][["SC_variant_id", "TVUR_shopify_rolling7days_lineitems_quantity"]].copy()
        forecasts_extreme_long_tail = forecasts_extreme_long_tail.merge(to_merge, how="left", on="SC_variant_id") 

        for q in ['0', '1', '2', '3', '4', '5', '6']:
                    forecasts_extreme_long_tail[q] = forecasts_extreme_long_tail["TVUR_shopify_rolling7days_lineitems_quantity"]
        forecasts_extreme_long_tail.drop(columns="TVUR_shopify_rolling7days_lineitems_quantity", inplace=True)
        first_date = forecasts_extreme_long_tail['TVKC_daydate'].iloc[0]
        forecasts_extreme_long_tail['model_date'] = first_date
        first_timestep = forecasts_extreme_long_tail['TVKR_time_idx'].iloc[0]
        forecasts_extreme_long_tail['timesteps'] = forecasts_extreme_long_tail['TVKR_time_idx'].apply(lambda x: x - first_timestep + 1)

        # Add the SC_model_type column with the value from self.model_type
        forecasts_extreme_long_tail['SC_model_type'] = model_type

        results = forecasts_extreme_long_tail.melt(id_vars=['SC_model_type', 'TVKR_time_idx',  'SC_product_category',  'SC_variant_id', 'TVKC_daydate', 'model_date', 'timesteps'],
                            value_vars=[str(i) for i in range(7)], 
                            var_name='quantile', 
                            value_name='forecast_values')

        results.sort_values(by=['TVKR_time_idx', 'quantile'], inplace=True)

        cols = results.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        results = results[cols]
        results = self.set_dataframe_datatypes(results)
        return results
    
    def normalize_attention(self) -> None:
        """
        Normalizes the 'attention' values in the model's interpretation.
        
        This method modifies the 'attention' values in the model's interpretation
        by performing several operations:

        1. It computes a cumulative sum of the 'encoder_length_histogram' values
        (excluding the first one) in reverse order, giving 'attention_occurrences'.
        2. It normalizes 'attention_occurrences' by dividing each of its values
        by its maximum value.
        3. It pads 'attention_occurrences' with ones to match the size of 'attention'.
        4. Finally, it divides each 'attention' value by the square of the corresponding
        'attention_occurrences' value, clamped to a minimum of 1.0. It then further
        normalizes 'attention' by dividing each of its values by the sum of all 'attention' values.

        The resulting 'attention' values are normalized and stored back in the interpreter.

        Benefits:
        The normalization of 'attention' values is a crucial step in the interpretation
        of the model's output. It ensures that the attention values are distributed
        in a standard range [0, 1], which makes these values easier to interpret and compare.
        Normalizing these values can also help identify important features in the input data
        that the model is paying more attention to. This can provide valuable insights
        into the model's decision-making process.
        """
        # Perform computations on attention_occurrences for normalization
        attention_occurrences = self.interpreter["encoder_length_histogram"][1:].flip(0).float().cumsum(0)
        attention_occurrences /= attention_occurrences.max()
        attention_occurrences = torch.cat(
            [
                attention_occurrences,
                torch.ones(
                    self.interpreter["attention"].size(0) - attention_occurrences.size(0),
                    dtype=attention_occurrences.dtype,
                    device=attention_occurrences.device,
                ),
            ],
            dim=0,
        )

        # Normalize the attention in the interpreter
        self.interpreter["attention"] /= attention_occurrences.pow(2).clamp(1.0)
        self.interpreter["attention"] /= self.interpreter["attention"].sum()
        logging.info("Normalized attention in the interpreter.")


    def create_interpretation_df(self) -> Dict[str, pd.DataFrame]:
        """
        Transforms the interpreter's data into a dictionary of pandas DataFrames.
        
        This function iterates through the interpreter's items, transforming them into pandas
        DataFrames which are stored in a dictionary with corresponding keys.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping each key in the interpreter to a DataFrame representation of its data.
        """
        df_dict = {}
        # Iterate through the interpreter
        for key, value in self.interpreter.items():
            try:
                # If the key is a DataFrame
                if key in ["attention", "static_variables", "encoder_variables", "decoder_variables"]:
                    # Create a new DataFrame from the interpreter
                    df = pd.DataFrame()
                    # Add the key to the DataFrame
                    value = value.detach().cpu().numpy()
                    # Normalize the values
                    value_np = value / value.sum()
                    # Add the values to the DataFrame
                    if key == "attention":
                        df["time_idx"] = range(1, len(value_np) + 1)
                    else:
                        # Add the values to the DataFrame
                        df['variables'] = getattr(self.model, key)
                    print("value_np",value_np.shape)
                    print("length of index",len(df.index))
                    # Change the normalised values to percent
                    df["importance_norm_percent"] = (value_np * 100).astype(np.float32)
                    # Add the model_date to the dictionary
                    df['model_date'] = pd.to_datetime(self.first_date)
                    # Add the model_type to the dictionary
                    df['model_type'] = self.model_type
                    df_dict[key] = df
            except Exception as e:
                logging.error(f"An error occurred while processing key {key}: {e}")
                traceback.print_exc()
                continue

        logging.info("Created DataFrames from the interpreter.")
        return df_dict



    def upload_plots_to_s3(self, figs: Dict[str, plt.Figure], s3_client, s3_bucket: str, s3_folder_path: str) -> None:
        """
        Uploads generated plots to an S3 bucket.

        This function iterates through the provided figures dictionary, saves each figure as a PNG file,
        and then uploads it to the specified S3 bucket and folder path. It cleans up by closing each plot after upload.

        Args:
            figs (Dict[str, plt.Figure]): A dictionary mapping names to matplotlib Figure objects.
            s3_client: An S3 client used to perform the upload.
            s3_bucket (str): The name of the S3 bucket to which plots should be uploaded.
            s3_folder_path (str): The path in the S3 bucket to which plots should be uploaded.
        """
        for fig_name in ['attention', 'static_variables', 'encoder_variables', 'decoder_variables']:
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                figs[fig_name].savefig(tmp.name)
                s3_client.upload_file(tmp.name, s3_bucket, os.path.join(s3_folder_path, f'{fig_name}_plot.png'))
                plt.close(figs[fig_name])
        logging.info("Uploaded plots to S3.")

    
    def generate_interpretation(self, s3_bucket: Optional[str] = None, s3_folder_path: Optional[str] = None, upload_plots: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Transforms the interpreter's data into a dictionary of pandas DataFrames.
        
        This function iterates through the interpreter's items, transforming them into pandas
        DataFrames which are stored in a dictionary with corresponding keys.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping each key in the interpreter to a DataFrame representation of its data.
        """
        try:
            if upload_plots and (s3_bucket is None or s3_folder_path is None):
                raise ValueError("s3_bucket and s3_folder_path must be provided if upload_plots is True.")
            
            # Initialize the S3 client
            try:
                s3_client = boto3.client('s3')
            except NoCredentialsError:
                logging.error("AWS credentials not found.")
                return
            except BotoCoreError as e:
                logging.error("An error occurred while connecting to AWS: ", e)
                return

            # Get the raw predictions
            try:
                self.raw_predictions, self.raw_x = self.model.predict(
                    self.new_prediction_data,
                    mode="raw",
                    return_x=True,
                )
            except Exception as e:
                logging.error(f"An error occurred while predicting: {e}")
                return

            # Create an interpreter instance
            try:
                self.interpreter = self.model.interpret_output(
                    out=self.raw_predictions,
                    reduction="sum"
                )
            except Exception as e:
                logging.error(f"An error occurred while interpreting the output: {e}")
                return


            # Normalize attention
            self.normalize_attention()
            try:
                # Create DataFrames from the interpreter
                interpretation_df_dict = self.create_interpretation_df()
            except Exception as e:
                logging.error(f"An error occurred while creating DataFrames: {e}")
                return

            # If upload_plots is True, generate and upload the plots
            if upload_plots:
                try:
                    figs = self.model.plot_interpretation(self.interpreter)
                    self.upload_plots_to_s3(figs, s3_client, s3_bucket, s3_folder_path)
                except Exception as e:
                    logging.error(f"An error occurred while uploading plots: {e}")
                    return

            logging.info("Generated interpretation and uploaded plots (if specified).")
            return interpretation_df_dict
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return

    def save_prediction_plots_to_s3(self, s3_bucket: str, s3_folder_path: str) -> None:
        """
        Save the generated prediction plots to the specified S3 path.

        Args:
            s3_bucket: The name of the S3 bucket.
            s3_folder_path: The folder path in the S3 bucket where the plots will be saved.

        Returns:
            None
        """
        # Initialize the S3 client
        s3_client = boto3.client('s3')

        # Get the raw predictions
        self.new_raw_predictions, self.new_raw_x = self.model.predict(
            self.new_prediction_data,
            mode="raw",
            return_x=True,
        )

        # Loop through the samples and generate the prediction plots
        for idx in range(len(self.x['encoder_lengths'])):
            # Generate the prediction plot for the current sample
            fig = self.model.plot_prediction(
                x=self.new_raw_x,
                out=self.new_raw_predictions,
                idx=idx,
                plot_attention=True,
                add_loss_to_title=True,
                show_future_observed=False
            )

            # Save the prediction plot to S3
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                fig.savefig(tmp.name)
                s3_client.upload_file(tmp.name, s3_bucket, os.path.join(s3_folder_path, f'prediction_plot_{idx}.png'))
                plt.close(fig)

