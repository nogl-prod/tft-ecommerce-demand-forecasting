import argparse
import os
import pickle
import copy
from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint # updated by Arda
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import boto3
import mlflow
import mlflow.pytorch

# Configure Garage for MLflow artifacts
os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://192.168.29.163:3900'))
os.environ.setdefault('AWS_ACCESS_KEY_ID', os.getenv('AWS_ACCESS_KEY_ID', ''))
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', os.getenv('AWS_SECRET_ACCESS_KEY', ''))
os.environ.setdefault('AWS_DEFAULT_REGION', os.getenv('AWS_DEFAULT_REGION', 'garage'))

# Set MLflow tracking URI
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.29.100:5000')
mlflow.set_tracking_uri(mlflow_tracking_uri)

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from sklearn.preprocessing._data import RobustScaler
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# from hyperparameter_config import *

import logging
from typing import Any, Dict, Tuple, Union

import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import optuna.logging
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import statsmodels.api as sm
from torch.utils.data import DataLoader

from pytorch_forecasting.metrics import QuantileLoss

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameter configuration
    parser.add_argument('--data_loader_batch_size', type = int, default = 6)
    parser.add_argument('--time_idx', type = str, default = 'TVKR_time_idx')
    parser.add_argument('--target', type = str, default = 'TVUR_shopify_lineitems_quantity')
    parser.add_argument('--group_ids', type = list, default = ['SC_variant_id'])
    parser.add_argument('--max_encoder_length', type = int, default = 380)
    parser.add_argument('--max_prediction_length', type = int, default = 90)
    parser.add_argument('--min_encoder_length', type = int, default = 190)
    parser.add_argument('--min_prediction_length', type = int, default = 1)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--input_data_bucket', type = str, default = 'nogl-welive-temp-storage-bucket')
    parser.add_argument('--input_file_path', type = str, default = 'training_data/consolidated.csv')
    parser.add_argument('--trainer_strategy', type = str, default = "ddp")
    parser.add_argument('--trainer_devices', type = int, default = -1)
    parser.add_argument('--trainer_gpus', type = int, default = -1)
    parser.add_argument('--trainer_max_epochs', type = int, default = 5)
    parser.add_argument('--tft_learning_rate', type = float, default = 0.01)
    parser.add_argument('--tft_hidden_size', type = int, default = 60)
    parser.add_argument('--tft_attention_head_size', type = int, default = 3)
    parser.add_argument('--tft_dropout', type = float, default = 0.15)
    parser.add_argument('--tft_hidden_continuous_size', type = int, default = 60)
    parser.add_argument('--trainer_gradient_clip_val', type = float, default = 0.01)
    parser.add_argument('--data_cutoff_earliest_date', type = str, default = "2021-01-01")
    parser.add_argument('--data_cutoff_latest_date', type = str, default = "2022-10-10")


    # I/O Paths
    parser.add_argument('--output-data-dir', type = str)
    parser.add_argument('--model-dir', type = str)
#     parser.add_argument('--train', type = str, default = os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--test', type = str, default = os.environ['SM_CHANNEL_TESTING'])

    # Configure Hosts
#     parser.add_argument('--hosts', type = list, default = json.loads(os.environ['SM_HOSTS']))
#     parser.add_argument('--current_host', type = str, default = json.loads(os.environ['SM_CURRENT_HOST']))

    return parser.parse_known_args()

 
def load_data(bucket, key):
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")


    print('Loading boto3 client')
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )
    
    
    print('Requesting S3 data')
    response = s3_client.get_object(Bucket=bucket, Key=key)

    print('Loading S3 Metadata')
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    print('Saving data to df')
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_csv(response.get("Body"))
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return df

def preprocess (args, data):
    
    print('Starting Preprocessing')
    data.columns = data.columns.str.replace(".", "_")
    print(data.columns[0:5])

    print('Loading Feature Types')
    static_categoricals = []
    static_reals = []
    time_varying_known_categoricals = []
    time_varying_known_reals = []
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = []
    
    for feature in data.columns:
        if feature.startswith("SC_"):
            static_categoricals.append(feature)
        if feature.startswith("SR_"):
            static_reals.append(feature)
        if feature.startswith("TVKC_"):
            time_varying_known_categoricals.append(feature)
        if feature.startswith("TVKR_"):
            time_varying_known_reals.append(feature)
        if feature.startswith("TVUC_"):
            time_varying_unknown_categoricals.append(feature)
        if feature.startswith("TVUR_"):
            time_varying_unknown_reals.append(feature)

    data['TVKC_daydate'] = pd.to_datetime(data['TVKC_daydate'])   
    data = data[data['TVKC_daydate'] > args.data_cutoff_earliest_date].copy()  
    data = data[data['TVKC_daydate'] < args.data_cutoff_latest_date].copy()
            

    print('Preprocessing Data')
    for col in (static_categoricals + time_varying_known_categoricals + time_varying_unknown_categoricals):
        data[col] = data[col].astype(str)
    for col in (static_reals + time_varying_known_reals + time_varying_unknown_reals):
        data[col] = data[col].astype("float")
   
        
    print('Setting Variable Groups')
    special_events=[
        'TVKC_external_holidays_and_special_events_by_date_external_importantSalesEvent',
        'TVKC_external_holidays_and_special_events_by_date_external_secondarySalesEvent',
        'TVKC_external_holidays_and_special_events_by_date_black_friday',
        'TVKC_external_holidays_and_special_events_by_date_cyber_monday',
        'TVKC_external_holidays_and_special_events_by_date_mothers_day',
        'TVKC_external_holidays_and_special_events_by_date_valentines_day',
        'TVKC_external_holidays_and_special_events_by_date_christmas_eve',
        'TVKC_external_holidays_and_special_events_by_date_fathers_day',
        'TVKC_external_holidays_and_special_events_by_date_orthodox_new_year',
        'TVKC_external_holidays_and_special_events_by_date_chinese_new_year',
        'TVKC_external_holidays_and_special_events_by_date_rosenmontag',
        'TVKC_external_holidays_and_special_events_by_date_carneval',
        'TVKC_external_holidays_and_special_events_by_date_start_of_ramadan',
        'TVKC_external_holidays_and_special_events_by_date_start_of_eurovision',
        'TVKC_external_holidays_and_special_events_by_date_halloween',
        'TVKC_external_holidays_and_special_events_by_date_saint_nicholas',
        'TVKC_external_holidays_and_special_events_by_date_external_holiday']

    for holiday_col in special_events:
        if holiday_col in time_varying_known_categoricals:
            time_varying_known_categoricals.remove(holiday_col)
            
    variable_groups = {
        'special_events': special_events
    }
    
    time_varying_known_categoricals.append('special_events')


    data[args.time_idx] = data[args.time_idx].astype("int")
    time_varying_known_reals.remove(args.time_idx)

    
    return data, static_categoricals, static_reals, time_varying_known_categoricals, time_varying_known_reals, time_varying_unknown_categoricals, time_varying_unknown_reals, special_events, variable_groups


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def model(args, train_dataloader, val_dataloader):
    
    print('START Training')
    
    # Set MLflow experiment
    client_name = getattr(args, 'client_name', 'default')
    mlflow.set_experiment(f"TFT-Training-{client_name}")
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            'learning_rate': args.tft_learning_rate,
            'hidden_size': args.tft_hidden_size,
            'dropout': args.tft_dropout,
            'max_epochs': args.trainer_max_epochs,
            'batch_size': args.data_loader_batch_size,
            'max_encoder_length': args.max_encoder_length,
            'max_prediction_length': args.max_prediction_length,
            'min_encoder_length': args.min_encoder_length,
            'min_prediction_length': args.min_prediction_length,
            'target': args.target,
            'client_name': client_name,
        })
        
        ### Updated by Arda ####
        from datetime import datetime, date, timedelta
        today = date.today()
        today = str(today).split("-")
        today = today[0] + today[1] + today[2]
        checkpoint_callback = ModelCheckpoint(
            save_top_k = 1,
            verbose = False,
            mode = "min",
            monitor = "val_loss",
            dirpath = os.getenv('CHECKPOINT_DIR', '/workspace/checkpoints/'),
            filename = today + "_tft-{epoch:02d}-{val_loss:.2f}"   
        )
        ########################
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
    
    #### comment the logger to figure out the error #####
    #logger = TensorBoardLogger(save_dir = args.output_data_dir, name = "lightning_logs")  # logging results to a tensorboard
    #####################################################
    
    print("Initialise TFT")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.tft_learning_rate,
        hidden_size=args.tft_hidden_size,
        attention_head_size=args.tft_attention_head_size,
        dropout=args.tft_dropout,
        hidden_continuous_size=args.tft_hidden_continuous_size,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4
    )
    
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    print("Initialise Trainer")
    trainer = pl.Trainer(
        max_epochs=args.trainer_max_epochs,
        strategy =args.trainer_strategy,
        devices=args.trainer_devices,
        gpus=args.trainer_gpus,
        accelerator = "gpu",
        #tpu_cores=0,
        gradient_clip_val=args.trainer_gradient_clip_val,
#         limit_train_batches=2,  # comment in for training, running validation every 30 batches TEST without
#         fast_dev_run=True,  # comment in to check that networkwork dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback], # updated by Arda
        #logger=logger,
    )
    
        print("Fit Training")
        trainer.fit(
            tft,
            train_dataloader,
            val_dataloader,
        )
        
        # Log metrics to MLflow
        best_val_loss = trainer.callback_metrics.get("val_loss")
        if best_val_loss:
            mlflow.log_metric("val_loss", best_val_loss.item())
        
        # Log model to MLflow (will save to Garage automatically!)
        mlflow.pytorch.log_model(
            tft,
            "model",
            registered_model_name=f"TFT-{client_name}"
        )
        
        # Log checkpoint path as artifact
        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(checkpoint_callback.best_model_path)
            mlflow.log_param("best_checkpoint_path", checkpoint_callback.best_model_path)
        
        print(f"Model logged to MLflow. Best checkpoint: {checkpoint_callback.best_model_path}")
    
    return trainer


if __name__ =='__main__':
    seed_everything(42, workers=True)
    
    print('########## Starting Main Program #########')
    args, unknown = parse_args()
    print('########## ARG : ', args)
    
    print('Loading Data')
    data = load_data(args.input_data_bucket,args.input_file_path) 
    data, static_categoricals, static_reals, time_varying_known_categoricals, time_varying_known_reals, time_varying_unknown_categoricals, time_varying_unknown_reals, special_events, variable_groups = preprocess (args, data)

    print('CREATING TIMESERIESDATASET')
    training = TimeSeriesDataSet(
        data, # only use week 0-138 for training
        time_idx=args.time_idx, # the rolling week counter as time_idx
        target=args.target, # defined above
        group_ids=args.group_ids, # defined above
        min_encoder_length=args.min_encoder_length,  # same as max_encoder_length as we always want to use 53 weeks
        max_encoder_length=args.max_encoder_length, # defined above
        min_prediction_length=args.min_prediction_length, #start with 1 as we want to predict 1-max_prediction_length in the future
        max_prediction_length=args.max_prediction_length, # defined above
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        variable_groups=variable_groups,  # group of categorical variables can be treated as one variable
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        categorical_encoders={"SC_variant_sku":NaNLabelEncoder(add_nan=True),
                          "TVKC_daydate": NaNLabelEncoder(add_nan=True),
                          "TVKC_year":NaNLabelEncoder(add_nan=True)},
        target_normalizer=GroupNormalizer(method='standard',
            groups=args.group_ids, transformation='log1p', center=False # need center=False to not have warning “scale is below 1e-7”
        ),  # use “softplus” and normalize by group TEST: try “log1p”
        #add_relative_time_idx=True, # for each sample relative time idx
        add_target_scales=True,
        #constant_fill_strategy=, #not used for now, maybe together with allow_missing_timesteps
        #allow_missing_timesteps=, #TEST: later with dropping -1 sellout_quantities, not used for now
        #lags=, #TEST: find out seasonalities and include
        scalers={"group_scaler":GroupNormalizer(method="robust", groups=args.group_ids)}, #TEST: if RobustScaler() better
        #categorical_encoders={"nan_label_encoder": NaNLabelEncoder(add_nan=True)},
    )

    print('CREATING VALIDATION DATA SET')
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    print('INITIALISING DATA LOADERS')
    train_dataloader = training.to_dataloader(train=True, 
                                          batch_size=args.data_loader_batch_size, 
                                          num_workers=args.num_workers, 
                                          #batch_sampler=batch_sampler, # TEST: with "synchronized"
#                                           batch_sampler="synchronized"
                                         )

    val_dataloader = validation.to_dataloader(train=False, 
                                              batch_size=args.data_loader_batch_size, 
                                              num_workers=args.num_workers,
                                              #batch_sampler=batch_sampler, # TEST: with "synchronized"
#                                               batch_sampler="synchronized"
                                             ) 
    
    forecast = model(args, train_dataloader, val_dataloader)    

    print('######## Best Trial ######')
#     print(f'BEST TRIAL PARAM: {forecast.best_trial.params}')
#     print(f'BEST TRIAL NUMBER: {forecast.best_trial.number}')
    
    print('######## Dumping Model ######')

#     joblib.dump(forecast, os.path.join(args.model_dir, "model.joblib"))
#     with open(os.path.join(args.model_dir, 'test_training_save'), 'wb') as f:
#         torch.save(forecast.state_dict(), f)
#     with open(os.path.join(args.model_dir, 'test_training_save.pickle'), 'wb') as f:
#         pickle.dump(forecast, f)
    
        


    
    #### TEST Remove once finished
    
    
    
    # Save the trained model to S3
    # with open(os.path.join(args.model_dir, 'test_study'), 'wb') as f:
    #     torch.save(study.state_dict(), f)
    
    # def optimize_hyperparameters(
    #     train_dataloaders: DataLoader,
    #     val_dataloaders: DataLoader,
    #     model_path: str,
    #     max_epochs: int = 20,
    #     n_trials: int = 100,
    #     timeout: float = 3600 * 8.0,  # 8 hours
    #     gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
    #     hidden_size_range: Tuple[int, int] = (16, 265),
    #     hidden_continuous_size_range: Tuple[int, int] = (8, 64),
    #     attention_head_size_range: Tuple[int, int] = (1, 4),
    #     dropout_range: Tuple[float, float] = (0.1, 0.3),
    #     learning_rate_range: Tuple[float, float] = (1e-5, 1.0),
    #     use_learning_rate_finder: bool = True,
    #     trainer_kwargs: Dict[str, Any] = {},
    #     log_dir: str = "lightning_logs",
    #     study: optuna.Study = None,
    #     verbose: Union[int, bool] = None,
    #     pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
    #     **kwargs,
    # ) -> optuna.Study:
    #     """
    #     Optimize Temporal Fusion Transformer hyperparameters.

    #     Run hyperparameter optimization. Learning rate for is determined with
    #     the PyTorch Lightning learning rate finder.

    #     Args:
    #         train_dataloaders (DataLoader): dataloader for training model
    #         val_dataloaders (DataLoader): dataloader for validating model
    #         model_path (str): folder to which model checkpoints are saved
    #         max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
    #         n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
    #         timeout (float, optional): Time in seconds after which training is stopped regardless of number of epochs
    #             or validation metric. Defaults to 3600*8.0.
    #         hidden_size_range (Tuple[int, int], optional): Minimum and maximum of ``hidden_size`` hyperparameter. Defaults
    #             to (16, 265).
    #         hidden_continuous_size_range (Tuple[int, int], optional):  Minimum and maximum of ``hidden_continuous_size``
    #             hyperparameter. Defaults to (8, 64).
    #         attention_head_size_range (Tuple[int, int], optional):  Minimum and maximum of ``attention_head_size``
    #             hyperparameter. Defaults to (1, 4).
    #         dropout_range (Tuple[float, float], optional):  Minimum and maximum of ``dropout`` hyperparameter. Defaults to
    #             (0.1, 0.3).
    #         learning_rate_range (Tuple[float, float], optional): Learning rate range. Defaults to (1e-5, 1.0).
    #         use_learning_rate_finder (bool): If to use learning rate finder or optimize as part of hyperparameters.
    #             Defaults to True.
    #         trainer_kwargs (Dict[str, Any], optional): Additional arguments to the
    #             `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ such
    #             as ``limit_train_batches``. Defaults to {}.
    #         log_dir (str, optional): Folder into which to log results for tensorboard. Defaults to "lightning_logs".
    #         study (optuna.Study, optional): study to resume. Will create new study by default.
    #         verbose (Union[int, bool]): level of verbosity.
    #             * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
    #             * 0 or False: log only warnings.
    #             * 1 or True: log pruning events.
    #             * 2: optuna logging level at debug level.
    #             Defaults to None.
    #         pruner (optuna.pruners.BasePruner, optional): The optuna pruner to use.
    #             Defaults to optuna.pruners.SuccessiveHalvingPruner().

    #         **kwargs: Additional arguments for the :py:class:`~TemporalFusionTransformer`.

    #     Returns:
    #         optuna.Study: optuna study results
    #     """

    #     print('Using Custom HPO')

    # #     from pytorch_forecasting.models.temporal_fusion_transformer.tuning import pl

    # #     class NewTrainer(pl.Trainer):
    # #         """
    # #         New trainer to remove the weights_summary
    # #         """
    # #         def __init__(self, *args, **kwargs):
    # #             del kwargs['weights_summary']
    # #             super().__init__(*args, **kwargs)
    # #     pl.Trainer = NewTrainer
    # #     print('Compatible Trainer Class Loaded')

    #     assert isinstance(train_dataloaders.dataset, TimeSeriesDataSet) and isinstance(
    #         val_dataloaders.dataset, TimeSeriesDataSet
    #     ), "dataloaders must be built from timeseriesdataset"

    #     logging_level = {
    #         None: optuna.logging.get_verbosity(),
    #         0: optuna.logging.WARNING,
    #         1: optuna.logging.INFO,
    #         2: optuna.logging.DEBUG,
    #     }
    #     optuna_verbose = logging_level[verbose]
    #     optuna.logging.set_verbosity(optuna_verbose)

    #     loss = kwargs.get(
    #         "loss", QuantileLoss()
    #     )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next

    #     # create objective function
    #     def objective(trial: optuna.Trial) -> float:


    #         # Filenames for each trial must be made unique in order to access each checkpoint.
    #         checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #             dirpath=os.path.join(model_path, "trial_{}".format(trial.number)), filename="{epoch}", monitor="val_loss"
    #         )

    #         # The default logger in PyTorch Lightning writes to event files to be consumed by
    #         # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    #         # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    #         metrics_callback = MetricsCallback()
    #         learning_rate_callback = LearningRateMonitor()
    #         logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
    #         gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
    #         default_trainer_kwargs = dict(             
    #             gpus=[0], #if torch.cuda.is_available() else None,
    #             accelerator='gpu',
    #             devices=4, 
    #             strategy="ddp_find_unused_parameters_false",
    #             max_epochs=max_epochs,
    #             gradient_clip_val=gradient_clip_val,
    #             callbacks=[
    #                 metrics_callback,
    #                 learning_rate_callback,
    #                 checkpoint_callback,
    #                 PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    #             ],
    #             logger=logger,
    #             enable_progress_bar=optuna_verbose < optuna.logging.INFO,
    #             enable_model_summary=[False, True][optuna_verbose < optuna.logging.INFO],
    #             deterministic=True,
    #         )
    #         default_trainer_kwargs.update(trainer_kwargs)
    #         trainer = pl.Trainer(
    #             **default_trainer_kwargs,
    #         )

    #         # create model
    #         hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
    #         kwargs["loss"] = copy.deepcopy(loss)
    #         model = TemporalFusionTransformer.from_dataset(
    #             train_dataloaders.dataset,
    #             dropout=trial.suggest_uniform("dropout", *dropout_range),
    #             hidden_size=hidden_size,
    #             hidden_continuous_size=trial.suggest_int(
    #                 "hidden_continuous_size",
    #                 hidden_continuous_size_range[0],
    #                 min(hidden_continuous_size_range[1], hidden_size),
    #                 log=True,
    #             ),
    #             attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
    #             log_interval=-1,
    #             **kwargs,
    #         )
    #         # find good learning rate
    #         if use_learning_rate_finder:
    #             lr_trainer = pl.Trainer(
    #                 gradient_clip_val=gradient_clip_val,
    # #                 gpus=2, # if torch.cuda.is_available() else None,
    #                 accelerator="gpu", 
    #                 devices=4, 
    #                 strategy="ddp_find_unused_parameters_false",
    #                 logger=False,
    #                 enable_progress_bar=False,
    #                 enable_model_summary=False,
    #                 deterministic=True,
    #             )
    #             res = lr_trainer.tuner.lr_find(
    #                 model,
    #                 train_dataloaders=train_dataloaders,
    #                 val_dataloaders=val_dataloaders,
    #                 early_stop_threshold=10000,
    #                 min_lr=learning_rate_range[0],
    #                 num_training=100,
    #                 max_lr=learning_rate_range[1],
    #             )

    #             loss_finite = np.isfinite(res.results["loss"])
    #             if loss_finite.sum() > 3:  # at least 3 valid values required for learning rate finder
    #                 lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
    #                     np.asarray(res.results["loss"])[loss_finite],
    #                     np.asarray(res.results["lr"])[loss_finite],
    #                     frac=1.0 / 10.0,
    #                 )[min(loss_finite.sum() - 3, 10) : -1].T
    #                 optimal_idx = np.gradient(loss_smoothed).argmin()
    #                 optimal_lr = lr_smoothed[optimal_idx]
    #             else:
    #                 optimal_idx = np.asarray(res.results["loss"]).argmin()
    #                 optimal_lr = res.results["lr"][optimal_idx]
    #             optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
    #             # add learning rate artificially
    #             model.hparams.learning_rate = trial.suggest_uniform("learning_rate", optimal_lr, optimal_lr)
    #         else:
    #             model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)

    #         # fit
    #         trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)

    #         # report result
    #         return metrics_callback.metrics[-1]["val_loss"].item()

    #     # setup optuna and run
    #     if study is None:
    #         study = optuna.create_study(direction="minimize", pruner=pruner)
    #     study.optimize(objective, n_trials=n_trials, timeout=timeout)
    #     return study