#!/usr/bin/env python
# coding: utf-8

# # 1. Imports

# ## 1.1 Exteral imports

import pytorch_forecasting


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

import copy
from pathlib import Path
import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing._data import RobustScaler
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from datetime import datetime, date, timedelta

# ## 1.2 Internal impors

############# DATE VARIABLES ##########################
today = date.today()
#today = today - timedelta(days=1) # delete it after testing 
yesterday = today - timedelta(days=1) # change back to days=1 , just to test training and inference now
today = str(today).split("-")
today = today[0] + today[1] + today[2]
########################################################

import boto3 
import sys
import os
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')

sys.path.append(src)

# import the 'config' funtion from the config.py file

from configAWSRDS import config

# from support functions:
from static_variables import *
from Support_Functions import *

import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument("--client_name", type=str, required=True)
parser.add_argument("--bundle", type=str, required=True)
# Parse the argument
args = parser.parse_args()

# get config params
section = args.client_name
print("client name:", section)
filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
params = config(section=section,filename=filename)
bundle = args.bundle
print(bundle)

# get model paths

def get_best_model(model_path):
    """
    gets best model from a directory 
    """
    best_loss = np.inf
    count = 0
    best_model = ""
    model_dirs = os.listdir(model_path)
    print(model_dirs)
    for i in model_dirs:
        date = i.split("_")
        date = date[0]
        if today == date:
            loss = i.split("=")
            loss = loss[-1].split(".")
            loss = loss[0] + "." + loss[1]
            loss = float(loss)
            if count == 0:
                best_loss = loss
            if loss <= best_loss:
                best_loss = loss
                best_model = model_path + i
            count += 1
        else:
            if len(model_dirs) == 1:
                print("Only one Model file is found and not up to date!!")
                best_model = model_path + i
            pass
    print("Best model:", best_model)
    return best_model

# no_rolling refers to top seller model and rolling refers to long tail model
# no_rolling refers to top seller model and rolling refers to long tail model
model_no_rolling_path = "/opt/ml/processing/input2/cross_client/checkpoints/no-rolling/"
model_rolling_path = "/opt/ml/processing/input2/cross_client/checkpoints/rolling/"

best_no_rolling = get_best_model(model_no_rolling_path)
best_rolling = get_best_model(model_rolling_path)

"""
# get best model for rolling
best_loss = np.inf
count = 0
for i in dirs_rolling:
    # model name : date_tft-{epoch:02d}-{val_loss:.2f}.ckpt
    date = i.split("_")
    date = date[0]
    if today == date:
        print(i)
        loss = i.split("=")
        loss = loss[-1].split(".")
        loss = loss[0] + "." + loss[1]
        loss = float(loss)
        if count == 0:
            best_loss = loss
        if loss <= best_loss:
            best_loss = loss
            best_rolling = model_rolling_path + i
        count += 1
    else:
        if len(dirs_rolling) == 1:
            print("Only one Model file is found and not up to date!!")
        pass

# get best model for no_rolling

best_loss = np.inf
count = 0
for i in dirs_no_rolling:
    # model name : date_tft-{epoch:02d}-{val_loss:.2f}.ckpt
    date = i.split("_")
    date = date[0]
    if today == date:
        loss = i.split("=")
        loss = loss[-1].split(".")
        loss = loss[0] + "." + loss[1]
        loss = float(loss)
        if count == 0:
            best_loss = loss
        if loss <= best_loss:
            best_loss = loss
            best_no_rolling = model_no_rolling_path + i
        count += 1
    else:
        if len(dirs_no_rolling) == 1:
            print("Only one Model file is found and not up to date!!")
        pass
"""

print(best_no_rolling)
print(best_rolling)

# create sqlalchemy connection
from sqlalchemy import create_engine

# ## 1.3 Options

# remove warnings
import warnings
warnings.filterwarnings('ignore')


# ## 1.4 Support functions


def prepare_data_for_inference(data, last_observation_day, max_prediction_length = 90, max_encoder_length = 380):
    
    # rename "." features to "_"
    for c in data.columns:
        if "." in c:
            print("Renaming:", c, " to:", c.replace(".","_"))
            data.rename(columns={c:c.replace(".","_")},inplace=True)
        
    # create category lists
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
            
    # adjust dtypes
    for col in (static_categoricals + time_varying_known_categoricals + time_varying_unknown_categoricals):
        data[col] = data[col].astype(str).astype("category")
    for col in (static_reals + time_varying_known_reals + time_varying_unknown_reals):
        data[col] = data[col].astype("float")
        
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

    #ADJUST to TVKR_time_idx (time_varying_known_reals.remove('TVKR_time_idx'))
    data['TVKR_time_idx'] = data['TVKR_time_idx'].astype("int")
    time_varying_known_reals.remove('TVKR_time_idx')
    
    # get time_idx for a specific date
    last_day_we_have_data = data[pd.to_datetime(data["TVKC_daydate"]) == pd.to_datetime(last_observation_day)].TVKR_time_idx.mean()
    if last_day_we_have_data.is_integer():
        last_day_we_have_data = int(last_day_we_have_data)
    else:
        print(last_day_we_have_data)
        raise ValueError("Different time_idx for daydates in data.")
    print("time_idx for last day of data is:", last_day_we_have_data)
    
    # limit time frame of data
    end_of_timeseries = last_day_we_have_data+max_prediction_length
    data_inf = data[lambda x: x["TVKR_time_idx"] <= end_of_timeseries]

    # calculate training cutoff
    training_cutoff = data_inf["TVKR_time_idx"].max() - max_prediction_length # in this case week 138
    print("training cutoff week:", training_cutoff)
    
    return data_inf, last_day_we_have_data, training_cutoff, end_of_timeseries


def get_results_featureimportance(model, data, first_prediction_time_idx, time_idx, max_encoder_length, max_prediction_length):
    # create data to predict on
    training_cutoff = first_prediction_time_idx - 1
    
    # select max_encoder_length of encoder data
    encoder_data = data[lambda x: (training_cutoff >= x[time_idx]) & (x[time_idx] > (training_cutoff-max_encoder_length))]

    # select max_prediction_length weeks of future data
    decoder_data = data[lambda x: (x[time_idx] > (training_cutoff))&(x[time_idx] < (first_prediction_time_idx+max_prediction_length))]
    
    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    new_raw_predictions, new_x = model.predict(new_prediction_data, mode="raw", return_x=True)
    interpretation = model.interpret_output(new_raw_predictions, reduction="sum")
    model.plot_interpretation(interpretation)
    
    return interpretation


def get_results(model, data, first_prediction_time_idx, time_idx, max_encoder_length, max_prediction_length):
    # create data to predict on
    training_cutoff = first_prediction_time_idx - 1
    
    # select max_encoder_length of encoder data
    encoder_data = data[lambda x: (training_cutoff >= x[time_idx]) & (x[time_idx] > (training_cutoff-max_encoder_length))]

    # select max_prediction_length weeks of future data
    decoder_data = data[lambda x: (x[time_idx] > (training_cutoff))&(x[time_idx] < (first_prediction_time_idx+max_prediction_length))]
    
    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    print(new_prediction_data)
    
    print("Encoder timeframe: ",encoder_data[time_idx].min(),"-",encoder_data[time_idx].max(), " length: ", len(encoder_data[time_idx].drop_duplicates()))
    print("Decoder timeframe: ",decoder_data[time_idx].min(),"-",decoder_data[time_idx].max(), " length: ", len(decoder_data[time_idx].drop_duplicates()))
    print("Prediction data timeframe: ",new_prediction_data[time_idx].min(),"-",new_prediction_data[time_idx].max(), " length: ", len(new_prediction_data[time_idx].drop_duplicates()))
    # create predictions
    
    predictions,x,index,decoder_lengths = model.predict(new_prediction_data,
                                                                       mode="quantiles",
                                                                       return_x=True,
                                                                       return_index = True,
                                                                       return_decoder_lengths=True)
                                                                       #add_nan=True)
    # build results data frame
    results = pd.DataFrame(columns=['SC_variant_id', 
                                    'q0',
                                    'q1',
                                    'q2',
                                    'q3',
                                    'q4',
                                    'q5',
                                    'q6'])

    # load data in new format
    for i in index.index:
        p = pd.DataFrame(predictions[i].numpy())
        #p[time_idx] = index.iloc[i,:][time_idx]
        p["SC_variant_id"] = index.iloc[i,:].SC_variant_id
        p.rename(columns={0:"q0",
                          1:"q1",
                          2:"q2",
                          3:"q3",
                          4:"q4",
                          5:"q5",
                          6:"q6"}, inplace=True)
        p = p[['SC_variant_id',
               'q0', 
               'q1', 
               'q2', 
               'q3', 
               'q4',
               'q5', 
               'q6']]
        
        
        results = pd.concat([results, p], ignore_index=True)
        
    # finish building time index
    time_idx_column = []
    for length in decoder_lengths:
        time_idx_column.extend(range(first_prediction_time_idx, first_prediction_time_idx + length))
    results[time_idx] = time_idx_column

    # rearrange column sorting
    cols = results.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    results = results[cols]
        
    return results



def load_TFT(path):
    best_tft_path = path
    device = torch.device('cpu')
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_tft_path, map_location=device)

    #disable randomness
    best_tft.eval()

    for i in [
    'hidden_size',
    'dropout',
    'hidden_continuous_size',
    'attention_head_size',
    'learning_rate']:
        print(i,best_tft.hparams.get(i))
    
    return best_tft


def lag_by_oneYear(df, column_name_to_lag, identifier):
    df = df.copy()
    df_newData = df.copy()
    df_newData["TVKC_daydate_plus1y"] = (pd.to_datetime(df_newData["TVKC_daydate"]).dt.date + 
                                         relativedelta(months = 12)).apply(lambda x: x.strftime("%Y-%m-%d"))
    df_newData = df_newData[[identifier,"TVKC_daydate_plus1y",column_name_to_lag]]
    df_newData = df_newData.groupby(["TVKC_daydate_plus1y",identifier], as_index=False).mean() # to eliminate duplicates in leap years 
                                                            # (e.g. 2020-02-29 -> will become 2021-02-28 so we have that double in 2021)
    df_newData.rename(columns={column_name_to_lag:column_name_to_lag+"_lastYear"}, inplace=True)
    df["TVKC_daydate"] = df["TVKC_daydate"].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
    #df.drop(columns="daydate_plus1y", inplace=True)
    df = df.merge(df_newData, how="left", left_on=[identifier, "TVKC_daydate"], right_on=[identifier, "TVKC_daydate_plus1y"])
    df.drop(columns="TVKC_daydate_plus1y", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=[identifier,"TVKC_daydate"], inplace=True)
    df[column_name_to_lag+"_lastYear"] = df[column_name_to_lag+"_lastYear"].fillna(method="ffill") # to fill nans 
                                                            # coming from 02-29 as they have no value from previous year
    return df


def year_week_edge_case(date, year, week):
    date = pd.to_datetime(date)
    print(date)
    year = int(year)
    week = int(week)
    if (date.strftime("%m") == "01") and ((week == 52) or (week == 53)):
        return str(year - 1)
    elif (date.strftime("%m") == "12") and (week == 1):
        return str(year + 1)
    else:
        return str(year)


# ## 2. Load models (MANUAL)

# load top seller TFT
# model without rolling
top_seller_TFT = load_TFT(best_no_rolling)
#top_seller_TFT = load_TFT("models/cuttoff-singledays-topAndLong-5epochs-2022-12-06-19-22-47-563/version_0/checkpoints/epoch=4-step=1820.ckpt")

# model with rolling
long_tail_TFT = load_TFT(best_rolling)


# ## 3. Load data (MANUAL)



# load data from .csv file
filename = "opt/ml/processing/input2/" + section + "/" + today + "_TopSeller_consolidated_cutoff.csv" # uncomment after testing is completed
#filename = "opt/ml/processing/input2/wefriends/20221227_TopSeller_consolidated_cutoff.csv"
data_top_seller = pd.read_csv(filename)
data_top_seller.drop(columns="Unnamed: 0", inplace=True)
#data_top_seller = data_top_seller[data_top_seller["SC_product_category_number"] == 8]


# load data from .csv file
filename = "opt/ml/processing/input2/" + section + "/" + today + "_LongTail_consolidated_cutoff.csv" # uncomment after testing is completed
#filename = "opt/ml/processing/input2/wefriends/20221227_LongTail_consolidated_cutoff.csv"
data_long_tail = pd.read_csv(filename)
data_long_tail.drop(columns="Unnamed: 0", inplace=True)
#data_long_tail = data_long_tail[data_long_tail["SC_product_category"] == "Leinwand"]


# load data from .csv file
filename = "opt/ml/processing/input2/" + section + "/" + today + "_Kicked_consolidated_cutoff.csv" # uncomment after testing is completed
#filename = "opt/ml/processing/input2/wefriends/20221227_Kicked_consolidated_cutoff.csv"
data_extreme_long_tail = pd.read_csv(filename)
data_extreme_long_tail.drop(columns="Unnamed: 0", inplace=True)
#data_long_tail = data_long_tail[data_long_tail["SC_product_category"] == "Leinwand"]


# # 4. Inference

# ## 4.1 Settings (MANUAL)

#last_observation_day =  str(yesterday) #"2022-12-28"  # should be yesterday
last_observation_day = "2023-01-01"
#last_observation_day = '2023-05-14'
max_prediction_length = 90
max_encoder_length = 380


# ## 4.1 Prepare data for forecasts

data_top_seller_inf, last_day_we_have_data_top_seller, training_cutoff_top_seller, end_of_timeseries_top_seller = prepare_data_for_inference(data_top_seller, 
                                                                                                                     last_observation_day = last_observation_day, 
                                                                                                                     max_prediction_length = max_prediction_length, 
                                                                                                                     max_encoder_length = max_encoder_length)




data_long_tail_inf, last_day_we_have_data_long_tail, training_cutoff_long_tail, end_of_timeseries_long_tail = prepare_data_for_inference(data_long_tail, 
                                                                                                                last_observation_day = last_observation_day, 
                                                                                                                max_prediction_length = max_prediction_length, 
                                                                                                                max_encoder_length = max_encoder_length)



data_extreme_long_tail_inf, last_day_we_have_data_extreme_long_tail, training_cutoff_extreme_long_tail, end_of_timeseries_extreme_long_tail = prepare_data_for_inference(data_extreme_long_tail, 
                                                                                                                                                last_observation_day = last_observation_day, 
                                                                                                                                                max_prediction_length = max_prediction_length, 
                                                                                                                                                max_encoder_length = max_encoder_length)




training_cutoff = 0
if (training_cutoff_top_seller != training_cutoff_long_tail) & (training_cutoff_top_seller != training_cutoff_extreme_long_tail):
    raise ValueError("Different training_cutoffs are not supported.")
else:
    training_cutoff = int((training_cutoff_top_seller+training_cutoff_long_tail+training_cutoff_extreme_long_tail)/3)
print("Training cutoff: ", training_cutoff)




end_of_timeseries = 0
if (end_of_timeseries_top_seller != end_of_timeseries_long_tail) & (end_of_timeseries_top_seller != end_of_timeseries_extreme_long_tail):
    raise ValueError("Different end_of_timeseries are not supported.")
else:
    end_of_timeseries = int((end_of_timeseries_top_seller+end_of_timeseries_long_tail+end_of_timeseries_extreme_long_tail)/3)
print("End of timeseries: ", end_of_timeseries)




last_day_we_have_data = 0
if (last_day_we_have_data_top_seller != last_day_we_have_data_long_tail) & (last_day_we_have_data_top_seller != last_day_we_have_data_extreme_long_tail):
    raise ValueError("Different training_cutoffs are not supported.")
else:
    last_day_we_have_data = int((last_day_we_have_data_top_seller+last_day_we_have_data_long_tail+last_day_we_have_data_extreme_long_tail)/3)
print("Last day we have data: ", last_day_we_have_data)


# ## 4.2 Get forecasts



# get top_seller forecasts
forecasts_top_seller = get_results(top_seller_TFT,
                                    data_top_seller_inf,
                                    first_prediction_time_idx=last_day_we_have_data_top_seller+1,
                                    time_idx="TVKR_time_idx",
                                    max_encoder_length=max_encoder_length,
                                    max_prediction_length=max_prediction_length)




forecasts_top_seller



# get long_tail forecasts
forecasts_long_tail = get_results(long_tail_TFT,
                                    data_long_tail_inf,
                                    first_prediction_time_idx=last_day_we_have_data_long_tail+1,
                                    time_idx="TVKR_time_idx",
                                    max_encoder_length=max_encoder_length,
                                    max_prediction_length=max_prediction_length)




forecasts_long_tail


# ### For extreme long tail



# create forecast df with only future time horizon
forecasts_extreme_long_tail = data_extreme_long_tail[(data_extreme_long_tail["TVKR_time_idx"] > last_day_we_have_data) & 
                                                          (data_extreme_long_tail["TVKR_time_idx"] <= end_of_timeseries)][["TVKR_time_idx", "SC_variant_id"]].copy()

# forecast extreme long tail with 7days average ffill by simply merging on only on SC_variant_id
to_merge = data_extreme_long_tail[data_extreme_long_tail["TVKR_time_idx"] == last_day_we_have_data][["SC_variant_id", "TVUR_shopify_rolling7days_lineitems_quantity"]].copy()
forecasts_extreme_long_tail = forecasts_extreme_long_tail.merge(to_merge, how="left", on="SC_variant_id")
# rename for compatability with other forecasts

for q in ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
    forecasts_extreme_long_tail[q] = forecasts_extreme_long_tail["TVUR_shopify_rolling7days_lineitems_quantity"]
forecasts_extreme_long_tail.drop(columns="TVUR_shopify_rolling7days_lineitems_quantity", inplace=True)
forecasts_extreme_long_tail


# ## 4.3 Merge data



forecasts = pd.concat([forecasts_top_seller, forecasts_long_tail, forecasts_extreme_long_tail])

print(forecasts.head())
# ## 4.4 Combine with original data


data_inf = pd.concat([data_top_seller_inf, data_long_tail_inf, data_extreme_long_tail_inf])




# get in pervious year sales
data_add = lag_by_oneYear(data_inf, "TVUR_shopify_lineitems_quantity", "SC_variant_id")




# cut data to forecast period
data_add = data_add[lambda x: x["TVKR_time_idx"] > training_cutoff]
data_add = data_add[lambda x: x["TVKR_time_idx"] <= end_of_timeseries]

data_add = data_add[['TVKC_daydate',
                    'TVKC_year',
                    'TVKC_month',
                    'TVKC_day',
                    'TVKC_weekday',
                    'TVKR_time_idx',
                    'SC_product_category_number',
                    'SC_product_category',
                    'SC_product_id',
                    'SC_variant_sku',
                    'SC_variant_inventory_item_id',
                    'SC_variant_id',
                    "TVKR_variant_RRP",
                    'TVKC_product_status',
                    'TVKC_variant_max_updated_at',
                    'TVUR_shopify_lineitems_quantity', # remove it after evaluate cross client
                    'TVKR_shopify_rolling30days_lineitems_quantity_lastYear',
                    'TVKR_shopify_rolling7days_lineitems_quantity_lastYear',
                    'TVUR_shopify_lineitems_quantity_lastYear',
                     "TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear",
                    "TVKR_shopify_rolling7days_lineitems_quantity_season_lastYear",
                     "TVKR_shopify_rolling7days_lineitems_quantity_resid_lastYear"]]

forecasts_consolidated = data_add.merge(forecasts, how="left", on=["TVKR_time_idx", 'SC_variant_id']).copy()
for i in range(7):
    forecasts_consolidated.rename(columns={"q"+str(i):"NOGL_forecast_q"+str(i)}, inplace=True)



# calculate potential revenue
for c in ["NOGL_forecast_q0", "NOGL_forecast_q1", "NOGL_forecast_q2", "NOGL_forecast_q3", "NOGL_forecast_q4", "NOGL_forecast_q5", "NOGL_forecast_q6"]:
    forecasts_consolidated[c+"_RRPrevenue"] = forecasts_consolidated["TVKR_variant_RRP"] * forecasts_consolidated[c]




top_seller = data_top_seller_inf.SC_variant_id.unique()
forecasts_consolidated["top_seller"] = forecasts_consolidated["SC_variant_id"].apply(lambda x: 1 if x in top_seller else 0)
forecasts_consolidated

# start here to comment for cross clinet


# # ## 4.5 Adjust forecasts statistically



# # assumption for market development
# market_change = 0.65


# # ### Load marketingAndSales_plan


# client = args.client_name
# DATASOURCE_PLAN = "/opt/ml/processing/input/Data Source Transformations/Plan Data/"
# filename = DATASOURCE_PLAN + client + "_Plan_Data_Weekly.xlsx"
# #plan_data_weekly_excel = "/opt/ml/processing/input/Data Source Transformations/Plan Data/Client01_Plan_Data_Weekly.xlsx"
# marketingAndSales_plan = pd.read_excel(filename, header=1)
# marketingAndSales_plan.rename(columns={"All":"planned_sales", "Week":"TVKC_week","Month":"TVKC_month","Quarter":"quarter","product_category":"SC_product_category"}, inplace=True)
# marketingAndSales_plan["TVKC_year"] = pd.to_datetime(marketingAndSales_plan["First Day of the Week"]).apply(lambda x: str(x.strftime("%Y")))
# marketingAndSales_plan


# # ### Calculate budget rations vs. last year



# category_budget_ratios = marketingAndSales_plan.copy()
# category_budget_ratios["total_budget"] = (category_budget_ratios["fb/insta_total_Budget"] + 
#                                           category_budget_ratios["googleads_total_Budget"] + 
#                                           category_budget_ratios["rest_total_budget"])

# # get only relevant time frame for forecast

# # this year
# category_budget_ratios_thisYear = category_budget_ratios[(category_budget_ratios["First Day of the Week"] > forecasts_consolidated.TVKC_daydate.min())&
#                                  (category_budget_ratios["First Day of the Week"] < forecasts_consolidated.TVKC_daydate.max())]

# # last year
# category_budget_ratios_lastYear = category_budget_ratios[(pd.to_datetime(category_budget_ratios["First Day of the Week"]) > pd.to_datetime(forecasts_consolidated.TVKC_daydate.min()) - relativedelta(days=365))&
#                                  (pd.to_datetime(category_budget_ratios["First Day of the Week"]) < pd.to_datetime(forecasts_consolidated.TVKC_daydate.max()) - relativedelta(days=365))]

# # combine again
# category_budget_ratios = pd.concat([category_budget_ratios_lastYear,category_budget_ratios_thisYear])


# # groupby and pivot
# category_budget_ratios = category_budget_ratios.groupby(["SC_product_category","TVKC_year"],as_index=False)["total_budget"].sum()
# category_budget_ratios = category_budget_ratios.pivot(index="SC_product_category", columns="TVKC_year")

# # flatten column index
# category_budget_ratios.columns = category_budget_ratios.columns.get_level_values(0) + '_' +  category_budget_ratios.columns.get_level_values(1)
# category_budget_ratios = category_budget_ratios.reset_index()

# category_budget_ratios


# # ### Calculate product ratios vs. last year


# data_product_ratios_lastYear = data_inf.copy()

# # only take data where sales are above 0
# data_product_ratios_lastYear = data_product_ratios_lastYear[data_product_ratios_lastYear["TVUR_shopify_lineitems_quantity"] > 0]

# # calculate number of products in 2021
# data_product_ratios_lastYear = data_product_ratios_lastYear[(data_product_ratios_lastYear["TVKR_time_idx"] > (forecasts_consolidated.TVKR_time_idx.min()-365))&
#                                                             (data_product_ratios_lastYear["TVKR_time_idx"] < (forecasts_consolidated.TVKR_time_idx.max()-365))].groupby("SC_product_category",as_index=False)["SC_variant_id"].nunique()

# data_product_ratios_lastYear.rename(columns={"SC_variant_id":"SC_variant_id_lastYear"}, inplace=True)

# # calculate number of products in 2022
# data_product_ratios_thisYear = data_inf.groupby("SC_product_category",as_index=False)["SC_variant_id"].nunique()
# data_product_ratios_thisYear.rename(columns={"SC_variant_id":"SC_variant_id_thisYear"}, inplace=True)

# product_ratios = data_product_ratios_thisYear.merge(data_product_ratios_lastYear, how="left", on="SC_product_category")

# product_ratios["SC_variant_id_lastYear"] = product_ratios["SC_variant_id_lastYear"].fillna(product_ratios["SC_variant_id_thisYear"])

# product_ratios


# # ### Combine cateogry and product to calculate marketing budget ratio



# # merge
# category_budget_ratios = category_budget_ratios.merge(product_ratios, how="left", on="SC_product_category").dropna()

# # make ratios proportional to products available

# # budget adjustments for last year
# category_budget_ratios["total_budget_2022"] = category_budget_ratios["total_budget_2022"]/category_budget_ratios["SC_variant_id_lastYear"]

# # budget adjustments for this year
# category_budget_ratios["total_budget_2023"] = category_budget_ratios["total_budget_2023"]/category_budget_ratios["SC_variant_id_thisYear"]



# # calculate ratio
# category_budget_ratios["budget_ratio"] = category_budget_ratios["total_budget_2023"] / category_budget_ratios["total_budget_2022"]

# # assuming not having done marketing before in that year we set factor to simply 1
# category_budget_ratios["budget_ratio"] = category_budget_ratios["budget_ratio"].replace([np.inf, -np.inf], 1)


# # ### Calculate standard deviation from trend_lastYear



# for_stats = data_inf[data_inf["TVKR_time_idx"] >= (training_cutoff - 365)]

# stdvs = for_stats.groupby("SC_variant_id", as_index=False)["TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear"].std()
# avgs = for_stats.groupby("SC_variant_id", as_index=False)["TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear"].mean()
# stdvs.rename(columns={"TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear":"stdv"}, inplace=True)
# avgs.rename(columns={"TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear":"avg"}, inplace=True)


# # ### Merge with forecasts_consoldiated


# # budget ratios
# forecasts_consolidated = forecasts_consolidated.merge(category_budget_ratios[["SC_product_category","budget_ratio"]], how="left", on="SC_product_category")

# # standard deviations for trend adjustments
# forecasts_consolidated = forecasts_consolidated.merge(stdvs[["SC_variant_id","stdv"]], how="left", on="SC_variant_id")
# forecasts_consolidated = forecasts_consolidated.merge(avgs[["SC_variant_id","avg"]], how="left", on="SC_variant_id")


# # ### Calculate adjusted forecasts



# # multiply forecasts by TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear/stdv * market_change * ratio

# for i in ['NOGL_forecast_q0',
#           'NOGL_forecast_q1',
#           'NOGL_forecast_q2',
#           'NOGL_forecast_q3',
#           'NOGL_forecast_q4',
#           'NOGL_forecast_q5',
#           'NOGL_forecast_q6']:
#     forecasts_consolidated[i+"_adjusted"] = (forecasts_consolidated[i] * 
#                                                              ((forecasts_consolidated["TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear"]-forecasts_consolidated["avg"])/forecasts_consolidated["stdv"]) *
#                                                                market_change
#                                                                  * forecasts_consolidated["budget_ratio"])



# # ## 4.6 Calculate Category Level Report

# # ### Combine forecasts on category level with marketingAndSales_plan

# print(forecasts_consolidated.head())

# # get isoweek for forecasts_consolidated
# forecasts_category = forecasts_consolidated.copy()

# forecasts_category["TVKC_week"] = pd.to_datetime(forecasts_category["TVKC_daydate"]).apply(lambda x: x.isocalendar()[1]) # --> (2022,40,1) 

# # adjust year_week_edge_cases
# forecasts_category["TVKC_year"] = forecasts_category.apply(lambda x: year_week_edge_case(x["TVKC_daydate"],x["TVKC_year"],x["TVKC_week"]), axis=1)

# # groupby forecasts
# forecasts_category = forecasts_category.groupby(["SC_product_category",
#                                                  "TVKC_year","TVKC_week"],
#                                                 as_index=False, 
#                                                 observed=True).agg({'SC_variant_id':"count",
#                                                                     "TVUR_shopify_lineitems_quantity_lastYear":"sum",
#                                                                     "TVKR_shopify_rolling7days_lineitems_quantity_trend_lastYear":"mean",
#                                                                     'NOGL_forecast_q1':"sum",
#                                                                     'NOGL_forecast_q2':"sum",
#                                                                     'NOGL_forecast_q3':"sum",
#                                                                     'NOGL_forecast_q4':"sum",
#                                                                     'NOGL_forecast_q5':"sum",
#                                                                     'NOGL_forecast_q1_adjusted':"sum",
#                                                                     'NOGL_forecast_q2_adjusted':"sum",
#                                                                     'NOGL_forecast_q3_adjusted':"sum",
#                                                                     'NOGL_forecast_q4_adjusted':"sum",
#                                                                     'NOGL_forecast_q5_adjusted':"sum",
#                                                                     "NOGL_forecast_q1_RRPrevenue":"sum", 
#                                                                     "NOGL_forecast_q2_RRPrevenue":"sum", 
#                                                                     "NOGL_forecast_q3_RRPrevenue":"sum", 
#                                                                     "NOGL_forecast_q4_RRPrevenue":"sum", 
#                                                                     "NOGL_forecast_q5_RRPrevenue":"sum"})

# forecasts_category.rename(columns={"SC_variant_id":"number_of_variants_included_in_that_categories_forecast"}, inplace=True)
# forecasts_category["number_of_variants_included_in_that_categories_forecast"] = forecasts_category["number_of_variants_included_in_that_categories_forecast"]/7

# # merge with marketingAndSales_plan
# forecasts_category = forecasts_category.merge(marketingAndSales_plan, how="left", on=["TVKC_year","TVKC_week","SC_product_category"])

# # calculate ROAS
# for c in ["NOGL_forecast_q1_RRPrevenue", "NOGL_forecast_q2_RRPrevenue", "NOGL_forecast_q3_RRPrevenue", "NOGL_forecast_q4_RRPrevenue", "NOGL_forecast_q5_RRPrevenue"]:
#     forecasts_category[forecasts_category["fb/insta_total_Budget"] == 0] = 999999999999999
#     forecasts_category[forecasts_category["googleads_total_Budget"] == 0] = 999999999999999
#     forecasts_category[forecasts_category["rest_total_budget"] == 0] = 999999999999999
    
#     forecasts_category[c+"_ROAS_FB"] = forecasts_category[c] / forecasts_category["fb/insta_total_Budget"]
#     forecasts_category[c+"_ROAS_Google"] = forecasts_category[c] / forecasts_category['googleads_total_Budget']
#     forecasts_category[c+"_ROAS_Rest"] = forecasts_category[c] / forecasts_category['rest_total_budget']
    
#     forecasts_category[forecasts_category["fb/insta_total_Budget"] == 999999999999999] = 0
#     forecasts_category[forecasts_category["googleads_total_Budget"] == 999999999999999] = 0
#     forecasts_category[forecasts_category["rest_total_budget"] == 999999999999999] = 0


# forecasts_category = forecasts_category[['SC_product_category',
#                                          'TVKC_year',
#                                          'quarter',
#                                          'TVKC_month',
#                                          'TVKC_week',
#                                          'First Day of the Week',
#                                          'number_of_variants_included_in_that_categories_forecast',
#                                          'planned_sales',
#                                          'TVUR_shopify_lineitems_quantity_lastYear',
#                                          'NOGL_forecast_q1',
#                                          'NOGL_forecast_q2',
#                                          'NOGL_forecast_q3',
#                                          'NOGL_forecast_q4',
#                                          'NOGL_forecast_q5',
#                                          #'NOGL_forecast_q1_adjusted',
#                                          #'NOGL_forecast_q2_adjusted',
#                                          #'NOGL_forecast_q3_adjusted',
#                                          #'NOGL_forecast_q4_adjusted',
#                                          #'NOGL_forecast_q5_adjusted',
#                                          "NOGL_forecast_q1_RRPrevenue",
#                                          "NOGL_forecast_q2_RRPrevenue",
#                                          "NOGL_forecast_q3_RRPrevenue",
#                                          "NOGL_forecast_q4_RRPrevenue",
#                                          "NOGL_forecast_q5_RRPrevenue",
#                                          'fb/insta_total_Budget',
#                                          'googleads_total_Budget',
#                                          'rest_total_budget',
#                                          'klaviyo_numberofcampaigns',
#                                          'klaviyo_grossreach_perweek',
#                                          'NOGL_forecast_q1_RRPrevenue_ROAS_FB',
#                                          'NOGL_forecast_q1_RRPrevenue_ROAS_Google',
#                                          'NOGL_forecast_q1_RRPrevenue_ROAS_Rest',
#                                          'NOGL_forecast_q2_RRPrevenue_ROAS_FB',
#                                          'NOGL_forecast_q2_RRPrevenue_ROAS_Google',
#                                          'NOGL_forecast_q2_RRPrevenue_ROAS_Rest',
#                                          'NOGL_forecast_q3_RRPrevenue_ROAS_FB',
#                                          'NOGL_forecast_q3_RRPrevenue_ROAS_Google',
#                                          'NOGL_forecast_q3_RRPrevenue_ROAS_Rest',
#                                          'NOGL_forecast_q4_RRPrevenue_ROAS_FB',
#                                          'NOGL_forecast_q4_RRPrevenue_ROAS_Google',
#                                          'NOGL_forecast_q4_RRPrevenue_ROAS_Rest',
#                                          'NOGL_forecast_q5_RRPrevenue_ROAS_FB',
#                                          'NOGL_forecast_q5_RRPrevenue_ROAS_Google',
#                                          'NOGL_forecast_q5_RRPrevenue_ROAS_Rest']]

# forecasts_category.head(5)

# removed

# ## 4.7 Interpret variable importances

# interpretation = get_results_featureimportance(long_tail_TFT,
#             data_long_tail_inf,
#             first_prediction_time_idx=last_day_we_have_data+1,
#             time_idx="TVKR_time_idx",
#             max_encoder_length=max_encoder_length,
#             max_prediction_length=max_prediction_length)

# ## 4.8 Product portfolio analysis


# load entire data


#shopify_sales.groupby("SC_variant_id",as_index=False)["TVUR_shopify_lineitems_quantity"].sum().groupby("TVUR_shopify_lineitems_quantity",as_index=False)["SC_variant_id"].nunique()


# # 5. Build dataframe incl. capa constraints for demand forecasting UI



engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True




# get sales data from shopify_sales
shopify_sales = import_data_AWSRDS(table="shopify_sales", schema="transformed", engine=engine)
shopify_products = import_data_AWSRDS(table="shopify_products", schema="transformed", engine=engine)



# change dtype of TVKC_daydate / daydate
forecasts_consolidated["TVKC_daydate"] = pd.to_datetime(forecasts_consolidated["TVKC_daydate"])
shopify_sales["daydate"] = pd.to_datetime(shopify_sales["daydate"])




# extract sales data from shopify_sales
forecasts_incl_history = shopify_sales[['product_category_number',
                                        'product_category',
                                        'product_id',
                                        'variant_sku',
                                        'variant_id',
                                        'daydate',
                                        'lineitems_quantity',
                                        'lineitems_price']].copy()

# calculate revenue historic and forecasted
forecasts_incl_history["revenue"] = forecasts_incl_history["lineitems_quantity"] * forecasts_incl_history["lineitems_price"] 

# merge on forecasts_consolidated
forecasts_incl_history = forecasts_incl_history.merge(forecasts_consolidated[['TVKC_daydate',
                                                                              'TVUR_shopify_lineitems_quantity',
                                                                              'SC_variant_id',
                                                                              'NOGL_forecast_q0',
                                                                              'NOGL_forecast_q1',
                                                                              'NOGL_forecast_q2',
                                                                              'NOGL_forecast_q3',
                                                                              'NOGL_forecast_q4',
                                                                              'NOGL_forecast_q5',
                                                                              'NOGL_forecast_q6',
                                                                              'NOGL_forecast_q0_RRPrevenue',
                                                                              'NOGL_forecast_q1_RRPrevenue',
                                                                              'NOGL_forecast_q2_RRPrevenue',
                                                                              'NOGL_forecast_q3_RRPrevenue',
                                                                              'NOGL_forecast_q4_RRPrevenue',
                                                                              'NOGL_forecast_q5_RRPrevenue',
                                                                              'NOGL_forecast_q6_RRPrevenue']].copy(), how="left", left_on=["variant_id", "daydate"], right_on=["SC_variant_id", "TVKC_daydate"])

# clean up
forecasts_incl_history.drop(columns=["TVKC_daydate", "SC_variant_id"], inplace=True)
forecasts_incl_history.fillna(0, inplace=True)


# add variant_title for better filtering expierence in Retool from shopify_products
forecasts_incl_history = forecasts_incl_history.merge(shopify_products[["variant_id", "variant_title"]].drop_duplicates(), how="left", on=["variant_id"])

# # 6. Export Excel & DB (MANUAL)

# ## AWS RDS
t = Timer("Upload")
forecasts_consolidated.to_sql("forecasts", con = engine, schema="forecasts", if_exists='replace', index=False, chunksize=1000, method="multi")
print("uploaded forecasts")
#forecasts_category.to_sql("forecasts_category", con = engine, schema="forecasts", if_exists='replace', index=False, chunksize=1000, method="multi")
#print("uploaded forecasts_category")

quantity_df = import_data_AWSRDS(schema="transformed",table="shopify_products",engine=engine)
quantity_df = quantity_df[["variant_sku","variant_inventory_quantity"]]

if bundle == "True":
    print("running if")
    if section == "cross_client":
        client_name = "stoertebekker"
    else:
        client_name = section
    forecasts_incl_history_and_bundling = unbundling(client_name, forecasts_incl_history, quantity_df)
    forecasts_incl_history_and_bundling = DIO(forecasts_incl_history_and_bundling)
    forecasts_incl_history_and_bundling.to_sql("forecasts_incl_history_and_bundling", con = engine, schema="forecasts", if_exists='replace', index=False, chunksize=1000, method="multi")
else:
    print("running else")
    forecasts_incl_history["isBundle"] = False
    forecasts_incl_history = forecasts_incl_history.merge(quantity_df, on="variant_sku")
    # select columns that contain 'forecast' in the column name
    forecast_cols = [col for col in forecasts_incl_history.columns if 'forecast' in col]
    # add new columns with suffix '_bundle' and "_total"
    for col in forecast_cols:
        forecasts_incl_history[col+'_bundle'] = 0
        forecasts_incl_history[col+'_total'] = forecasts_incl_history[col]

    forecasts_incl_history = DIO(forecasts_incl_history)

    forecasts_incl_history.to_sql("forecasts_incl_history_and_bundling", con = engine, schema="forecasts", if_exists='replace', index=False, chunksize=1000, method="multi")
t.end()

engine.dispose()
