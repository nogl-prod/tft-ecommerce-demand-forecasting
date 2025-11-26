import os
import logging
from typing import Dict, Tuple, Optional
from datetime import date

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import boto3
import mlflow

# Configure Garage for MLflow artifacts
os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://192.168.29.163:3900'))
os.environ.setdefault('AWS_ACCESS_KEY_ID', os.getenv('AWS_ACCESS_KEY_ID', ''))
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', os.getenv('AWS_SECRET_ACCESS_KEY', ''))
os.environ.setdefault('AWS_DEFAULT_REGION', os.getenv('AWS_DEFAULT_REGION', 'garage'))

# Set MLflow tracking URI (defaults to environment variable or localhost)
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.29.100:5000')
mlflow.set_tracking_uri(mlflow_tracking_uri)

from tempfile import NamedTemporaryFile
# Set up logging to output to console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)


class TFTModelHandler:
    """
    A class for handling Temporal Fusion Transformer (TFT) models, allowing users to load the best model from a directory
    based on a given metric and optional date string.

    Example usage:
    ```
    # Import the necessary libraries
    from tft_model_handler import TFTModelHandler

    # Define model directory and desired metric
    MODEL_DIRECTORY = "/path/to/your/model/directory"
    METRIC = "loss"

    # Create a TFTModelHandler instance
    model_handler = TFTModelHandler(MODEL_DIRECTORY)

    # Load the best model based on the specified metric
    best_model = model_handler.load_best_model()
    ```

    Attributes:
    - model_path (str): Path to the directory containing the model checkpoints.
    - metric (str): Metric to use for selecting the best model. Defaults to "loss".
    - seed (int): Value to use as the random seed. Defaults to 42.
    - device (torch.device): Device to load the model on. Set to GPU if available, otherwise CPU.
    - date_str (Optional[str]): Date string in the format YYYYMMDD. If provided, only considers model checkpoints created on this date.
    """
    def __init__(
        self,
        model_path: str,
        metric: Optional[str] = "loss",
        seed: Optional[int] = 42,
        date_str: Optional[str] = None,
    ) -> None:
        """
        Initializes TFTModelHandler class.

        Args:
        - model_path (str): Path to the directory containing the model checkpoints.
        - metric (str, optional): Metric to use for selecting the best model. Defaults to "loss".
        - seed (int, optional): Value to use as the random seed. Defaults to 42.
        - date_str (str, optional): Date string in the format YYYYMMDD. If provided, only considers model checkpoints created on this date.

        Raises:
        - NotADirectoryError: If the model_path is not a directory.
        - ValueError: If the metric string is not "loss" or "mean_absolute_error".
        """
        self.model_path = model_path
                
        if 'rolling' in self.model_path:
            if 'no-rolling' in self.model_path:
                self.model_type = 'top_seller'
            else:
                self.model_type = 'long_tail'
        else:
            self.model_type = 'extreme_long_tail'
                   
        self.metric = metric
        logging.info(f"Model type: {self.model_type}")
        self.seed = seed

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.date_str = date_str

        logging.info(
            f"TFTModelHandler initialized with model path: {model_path}, metric: {metric}, seed: {seed}"
        )

    @staticmethod
    def get_today() -> str:
        """
        Returns the current date in the format YYYYMMDD.

        Returns:
        - today (str): Current date in the format YYYYMMDD.
        """
        return date.today().strftime("%Y%m%d")

    def load_specific_checkpoint_from_s3(self, model_path: str = None) -> TemporalFusionTransformer:
        """
        Loads a TemporalFusionTransformer model from a checkpoint file.

        Args:
        - path (str): Path to the checkpoint file.

        Returns:
        - tft_model (TemporalFusionTransformer): Loaded TFT model.
        """
        if model_path:
            self.model_path = model_path
        # Set up AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "Missing required AWS credentials. Please set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables. See .env.example for details."
            )
        
        # Create a custom boto3 session with the credentials and region
        boto3_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )

        # Create a boto3 client with the custom boto3 session
        s3_client = boto3_session.client('s3')

        # Parse the S3 path
        s3_parts = self.model_path.replace("s3://", "").split("/")
        logging.info(f"Parsed S3 path: {s3_parts}")
        bucket_name = s3_parts[0]
        logging.info(f"Parsed S3 bucket name: {bucket_name}")
        object_key = "/".join(s3_parts[1:])
        logging.info(f"Parsed S3 object key: {object_key}")

        # Download the checkpoint file to a temporary file
        with NamedTemporaryFile(suffix=".ckpt") as tmp_file:
            s3_client.download_file(bucket_name, object_key, tmp_file.name)
            logging.info(f"Loading TFT model from S3 checkpoint file: {self.model_path}")

            # Load TFT model from checkpoint file
            tft_model = TemporalFusionTransformer.load_from_checkpoint(
                tmp_file.name, map_location=self.device
            )

        # Disable randomness
        torch.manual_seed(self.seed)
        tft_model.eval()

        # Log selected model hyperparameters
        hyperparams = {
            i: tft_model.hparams.get(i)
            for i in [
                "hidden_size",
                "dropout",
                "hidden_continuous_size",
                "attention_head_size",
                "learning_rate",
            ]
        }
        logging.info(f"TFT model loaded with hyperparameters: {hyperparams}")
        # tft_model = mlflow.pytorch.load_model(self.model_path,map_location=self.device)
        #  Log selected model hyperparameters
        # hyperparams = {
        #     i: tft_model.hparams.get(i)
        #     for i in [
        #         "hidden_size",
        #         "dropout",
        #         "hidden_continuous_size",
        #         "attention_head_size",
        #         "learning_rate",
        #     ]
        # }
        # logging.info(f"TFT model loaded with hyperparameters: {hyperparams}")
        return tft_model, self.model_type

    def load_specific_checkpoint(self,model_path: str=None) -> TemporalFusionTransformer:
        """
        Loads a TemporalFusionTransformer model from a checkpoint file.

        Args:
        - path (str): Path to the checkpoint file.

        Returns:
        - tft_model (TemporalFusionTransformer): Loaded TFT model.
        """
        if model_path:
            self.model_path = model_path
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"{self.model_path} is not a file.")
        logging.info(f"Loading TFT model from checkpoint file: {self.model_path}")

        # Load TFT model from checkpoint file
        tft_model = TemporalFusionTransformer.load_from_checkpoint(
            self.model_path, map_location=self.device
        )

        # Disable randomness
        torch.manual_seed(self.seed)
        tft_model.eval()

        # Log selected model hyperparameters
        hyperparams = {
            i: tft_model.hparams.get(i)
            for i in [
                "hidden_size",
                "dropout",
                "hidden_continuous_size",
                "attention_head_size",
                "learning_rate",
            ]
        }
        logging.info(f"TFT model loaded with hyperparameters: {hyperparams}")

        return tft_model, self.model_type

    def get_best_model(self) -> str:
        """
        Finds the best model checkpoint file in the model directory based on a given metric.

        Returns:
        - best_model (str): Path to the best model checkpoint file.
        """
        logging.info(f"Finding best model based on metric: {self.metric}")

        model_files = os.listdir(self.model_path)
        if not model_files:
            # If the model directory does not exist, raise an exception
            raise NotADirectoryError(f"Model directory not found at {self.model_path}")
        logging.info(f"Model directory found at {self.model_path}")

        best_value = np.inf
        best_model_path = ""

        for file in model_files:
            if not file.endswith(".ckpt"):
                continue
            # Extract the date and metric value from the filename
            parts = file.split("_")
            if len(parts) < 3 or parts[-2] != self.metric + "=":
                continue
            date_str = parts[0]
            try:
                value = float(parts[-1].rstrip(".ckpt"))
            except ValueError:
                # If the split fails or metric value is not a number, skip this file
                continue
            if self.date_str is not None and self.date_str != date_str:
                # If provided date string doesn't match the date string in this checkpoint file name, skip this file
                continue
            if value < best_value:
                # If this file has a better metric value than the current best, update the best
                best_value = value
                best_model_path = os.path.join(self.model_path, file)
        if not best_model_path:
            # If no suitable model is found based on the given metric and date, raise an exception
            raise FileNotFoundError(
                f"No suitable model checkpoint files found in '{self.model_path}'."
            )
        logging.info(f"Best TFT model found at: {best_model_path}")
        return best_model_path

        
    def load_best_model(self) -> Tuple[TemporalFusionTransformer, Dict[str, float]]:
        """
        Loads the best TFT model based on a given metric.

        Returns:
        - best_model (TemporalFusionTransformer): Loaded best TFT model.
        """

        try:
            best_model_path = self.get_best_model()
            best_model = self.load_specific_checkpoint(best_model_path)
            return best_model, self.model_type
        except Exception as e:
            # If there is an exception, print traceback and re-raise the exception
            logging.exception(f"Failed to load best TFT model with exception: {e}")
            raise e
        
    def create_model(
        self,
        training_data,
        params
    ) -> TemporalFusionTransformer:
        """
        Creates TFT model
        Returns:
            TFT model
        """
        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=params.learning_rate,
            hidden_size=params.hidden_size,
            attention_head_size=params.attention_head_size,
            dropout=params.dropout,
            hidden_continuous_size=params.hidden_continuous_size,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4
        )
        return tft
    
    def create_trainer(
        self,
        params      
    ) -> pl.Trainer :
        
        # Initialize callbacks

        from datetime import datetime, date, timedelta
        model_date = params.model_date
        model_date = str(model_date).split("-")
        model_date = model_date[0] + model_date[1] + model_date[2]
        checkpoint_callback = ModelCheckpoint(
            save_top_k = 1,
            verbose = False,
            mode = "min",
            monitor = "val_loss",
            dirpath = "/opt/ml/checkpoints/",
            filename = model_date + "_tft-{epoch:02d}-{val_loss:.2f}"   
        )
        ########################
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=params.epoch_num, #self.args.trainer_max_epochs,
            strategy = params.data_strategy, #self.args.trainer_strategy,
            devices=params.num_devices,
            gpus=params.num_devices,
            accelerator = "gpu",
            gradient_clip_val=params.gradient_clip, #self.args.trainer_gradient_clip_val,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        )
        
        return trainer
    

        



# Example usage:
# model_handler = TFTModelHandler(MODEL_DIRECTORY)
# best_model = model_handler.load_best_model(metric=METRIC)

"""
Inline documentation:

- The code imports necessary libraries for the script to run.
- The code defines a class TFTModelHandler that contains methods for loading and selecting the best TemporalFusionTransformer model.
- The _init_ method initializes the class with a model_path argument that specifies the directory containing the model files.
- The _get_today method returns the current date in the format YYYYMMDD.
- The load_TFT method loads a TemporalFusionTransformer model from a checkpoint file specified by the path argument. It returns the loaded TFT model and a dictionary containing the selected model hyperparameters.
- The get_best_model method finds the best model file in the model directory based on a given metric. It returns the path to the best model file.
- The load_best_model method loads the best model based on a given metric. It returns the loaded best TFT model.
- The example usage section shows how to use the TFTModelHandler class to load the best model and its hyperparameters.
"""
