# # 1. Imports/Options/Support functions

# ## 1.1 External imports

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import os
import time
import math
from sqlalchemy import create_engine
import gc
import multiprocessing
import dask.dataframe as dd

#import memory_profiler

# ## 1.2 Internal imports

# set path
import sys
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)

# import the 'config' funtion from the config.py file
from configAWSRDS import config

# from support functions:
from Support_Functions import *

# ## 1.3 Options

# no max columns for pandas dataframe for a better visualization on jupyter notebooks
pd.set_option('display.max_columns', None)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# # 1.4 Define / Import Variables

# set date
date = str(datetime.now().strftime("%Y%m%d"))

# argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--client_name", type=str, required=True)
args = parser.parse_args()

# get config params
section = args.client_name # section = client_name
print("client name:", section)
filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
params = config(section=section,filename=filename)

# ## 1.5 Support functions

date = str(datetime.now().strftime("%Y%m%d"))

class Timer():
    """
    This class allows to measure the time.
    When initiated it prints a message given in the constructor and prints the elapsed time
    after on end().
    """
    def __init__(self,message):
        print(message,end="... ")
        self.start_time = time.time()

    def end(self):
        stop_time = time.time()
        total = round(stop_time - self.start_time,1)
        print("done",str(total)+"s")

def rename_columns(df, sceleton, prefix = "", suffix = "", list_of_columns=[], all_columns=True):
    """
    Renames all the columns with given prefix or suffix if all_columns=True.
    Renames all given columns in list_of_columns with given prefix or suffix if all_columns = False and list_of_columns is not empty.
    DISCLAIMER: variant_id and daydate cannot be renamed.
    """
    if all_columns==True:
        df = df.add_prefix(prefix)
        df = df.add_suffix(suffix)
    if (all_columns==False) & (len(list_of_columns)) > 0:
        for c in list_of_columns:
            df.rename(columns={c:prefix+c}, inplace=True)
            df.rename(columns={c:c+suffix}, inplace=True)
    for c in sceleton.columns:
        df.rename(columns={prefix+c+suffix:c}, inplace=True)
    return df

def merge_with_productsxdaysscelecton(base, all_dfs_to_be_merged, keys):
    df_merged = base.copy()
    number_of_columns = len(list(base.columns))
    total_number_of_columns = len(list(base.columns))
    number_of_merging_keys = len(keys)
    for df in all_dfs_to_be_merged:
        text = "Merging dataframe with " + str(len(list(df.columns))-number_of_merging_keys) + " columns"
        t = Timer(text)
        df_merged = df_merged.merge(df, how = "left", on=keys)
        t.end()
        print(df_merged.shape)
        total_number_of_columns = total_number_of_columns + (len(list(df.columns))-number_of_merging_keys)
    print(total_number_of_columns)
    return df_merged

def get_minmax_above0_on_feature_level(df, minmaxfeature):

    # create results dataframe
    df_results = pd.DataFrame(list(df.columns), columns=["features"])
    df_results["min"] = ""
    df_results["max"] = ""
    # loop over dataframe with only minmaxfeature and feature name
    for c in list(df.columns):
        # only minmaxfeature and feature name 
        df_feature = df[[minmaxfeature, c]]
        # filter out all above 0
        df_feature = df_feature[df_feature[c] != 0]
        # get max and min date
        maxiumum = df_feature[minmaxfeature].max()
        minimum = df_feature[minmaxfeature].min()
        df_toadd = pd.DataFrame([[c, minimum,maxiumum]], columns=["features","minimum","maximum"])
        df_results = df_results.join(df_toadd.set_index(["features"]), on=["features"])
        df_results.loc[df_results.minimum.notna(), 'min'] = df_results.minimum
        df_results = df_results.drop(columns="minimum")
        df_results.loc[df_results.maximum.notna(), 'max'] = df_results.maximum
        df_results = df_results.drop(columns="maximum")
    
    # transform to str for grouping
    df_results["min"] = df_results["min"].astype("str")
    df_results["max"] = df_results["max"].astype("str")

    # add nan and zero counts to df_results
    nan_counts = df.isna().sum()
    zero_counts = (df == 0).sum()
    total_counts = len(df)

    nan_percentages = (nan_counts / total_counts) * 100
    zero_percentages = (zero_counts / total_counts) * 100

    result_df = pd.DataFrame({
        'nan_percentage': nan_percentages,
        'zero_percentage': zero_percentages
    })

    result_df.reset_index(inplace=True)
    result_df.rename(columns={"index":"features"}, inplace=True)

    df_results = df_results.merge(result_df, how="left", on="features")

    # Create an empty DataFrame to store the results
    results = pd.DataFrame()

    # Process each column name from the original DataFrame
    for column_name in df.columns:
        # Split the column name into substrings using "_"
        substrings = column_name.split("_")

        # Create a dictionary to store the substrings as key-value pairs
        substring_dict = {}

        # Assign the substrings to the dictionary keys
        for i, substring in enumerate(substrings, start=1):
            key = f"substring_{i}"
            substring_dict[key] = substring

        # Convert the dictionary to a DataFrame row
        row_df = pd.DataFrame(substring_dict, index=[column_name])

        # Append the row DataFrame to the result DataFrame
        results = pd.concat([results, row_df])

    results.reset_index(inplace=True)
    results.rename(columns={"index":"features"}, inplace=True)

    df_results = df_results.merge(results, how="left", on="features")
    
    # create min and max analysis data frame
    df_results_min_analysis = pd.DataFrame(df_results.groupby("min")["features"].apply(list)).merge(pd.DataFrame(df_results.groupby("min")["features"].count()), on="min", how="left")
    df_results_min_analysis.rename(columns={"features_x":"feature_names", "features_y":"counts"}, inplace=True)
    df_results_min_analysis.reset_index(inplace=True)
    df_results_max_analysis = pd.DataFrame(df_results.groupby("max")["features"].apply(list)).merge(pd.DataFrame(df_results.groupby("max")["features"].count()), on="max", how="left")
    df_results_max_analysis.rename(columns={"features_x":"feature_names", "features_y":"counts"}, inplace=True)
    df_results_max_analysis.reset_index(inplace=True)
    
    # count number of features in the variable categories (SC, SR, TVKC, TVKR, TVUC, TVKR)
    
    # add counting columns to max and min analysis
    for n in ["count_SC",
              "count_SR",
              "count_TVKC",
              "count_TVKR",
              "count_TVUC",
              "count_TVUR"]:
        df_results_min_analysis[n] = 0
        df_results_max_analysis[n] = 0
    
    # add counts to min analysis
    for i in range(len(list(df_results_min_analysis.feature_names))):
        count_SC = 0
        count_SR = 0
        count_TVKC = 0 
        count_TVKR = 0
        count_TVUC = 0
        count_TVUR = 0
        
        to_count = list(df_results_min_analysis.feature_names)[i]
        
        for e in to_count:
            if e.startswith("SC_"):
                count_SC += 1
            if e.startswith("SR_"):
                count_SR += 1
            if e.startswith("TVKC_"):
                count_TVKC += 1
            if e.startswith("TVKR_"):
                count_TVKR += 1
            if e.startswith("TVUC_"):
                count_TVUC += 1
            if e.startswith("TVUR_"):
                count_TVUR += 1
                
        df_results_min_analysis["count_SC"].iloc[i] = count_SC
        df_results_min_analysis["count_SR"].iloc[i] = count_SR
        df_results_min_analysis["count_TVKC"].iloc[i] = count_TVKC
        df_results_min_analysis["count_TVKR"].iloc[i] = count_TVKR
        df_results_min_analysis["count_TVUC"].iloc[i] = count_TVUC
        df_results_min_analysis["count_TVUR"].iloc[i] = count_TVUR
        
    # add counts to max analysis
    for i in range(len(list(df_results_max_analysis.feature_names))):
        count_SC = 0
        count_SR = 0
        count_TVKC = 0 
        count_TVKR = 0
        count_TVUC = 0
        count_TVUR = 0
        
        to_count = list(df_results_max_analysis.feature_names)[i]
        
        for e in to_count:
            if e.startswith("SC_"):
                count_SC += 1
            if e.startswith("SR_"):
                count_SR += 1
            if e.startswith("TVKC_"):
                count_TVKC += 1
            if e.startswith("TVKR_"):
                count_TVKR += 1
            if e.startswith("TVUC_"):
                count_TVUC += 1
            if e.startswith("TVUR_"):
                count_TVUR += 1
                
        df_results_max_analysis["count_SC"].iloc[i] = count_SC
        df_results_max_analysis["count_SR"].iloc[i] = count_SR
        df_results_max_analysis["count_TVKC"].iloc[i] = count_TVKC
        df_results_max_analysis["count_TVKR"].iloc[i] = count_TVKR
        df_results_max_analysis["count_TVUC"].iloc[i] = count_TVUC
        df_results_max_analysis["count_TVUR"].iloc[i] = count_TVUR
        
    # save to results df
    return df_results, df_results_min_analysis, df_results_max_analysis

def cutoff_beginning_of_timeseries(data, target, time_idx, identifier):
    data_after = pd.DataFrame(columns=data.head(0).columns)
    for p in data[identifier].unique():
        data_sub = data[data[identifier] == p].copy()
        # find first point higher 0
        cutoff = data_sub[(data_sub[target] > 0)][time_idx].min() - 1
        if math.isnan(cutoff) == True:
            cutoff = 0
        print("Cutoff for", p,"is:", cutoff, "Sum of sales is:", data_sub[target].sum())
        data_sub = data_sub[(data[time_idx] > cutoff)]
        data_after = pd.concat([data_after,data_sub])
    return data_after

def distinguish_features(df,
                         sceleton,
                         static_categoricals,
                         static_reals,
                         time_varying_known_categoricals,
                         time_varying_known_reals,
                         time_varying_unknown_categoricals):
                         #time_varying_unknown_reals
    for c in df.columns:
        if c == "daydate":
            df.rename(columns={"daydate":"TVKC_daydate"}, inplace=True)
        elif c == "variant_id":
            df.rename(columns={"variant_id":"SC_variant_id"}, inplace=True)
        elif c in static_categoricals:
            df.rename(columns={c:"SC_"+c}, inplace=True)
        elif c in static_reals:
            df.rename(columns={c:"SR_"+c}, inplace=True)
        elif c in time_varying_known_categoricals:
            df.rename(columns={c:"TVKC_"+c}, inplace=True)
        elif c in time_varying_known_reals:
            df.rename(columns={c:"TVKR_"+c}, inplace=True)
        elif c in time_varying_unknown_categoricals:
            df.rename(columns={c:"TVUC_"+c}, inplace=True)
        else:
            df.rename(columns={c:"TVUR_"+c}, inplace=True)
    return df

def get_dtypes(df, sceleton):
    dtype_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"feature", 0:"dtype"})
    for c in df.columns:
        if c in list(sceleton.columns):
            dtype_df = dtype_df[dtype_df.feature != c]
    dtype_df.reset_index(drop=True, inplace=True)
    return dtype_df

def add_time_idx(df, timeseries_ids=["SC_variant_id", "TVKC_daydate"], date_id="TVKC_daydate"):
    """
    Adds a time index column to the DataFrame based on the specified timeseries IDs and date ID.

    Args:
        df (pandas.DataFrame): The DataFrame to which the time index will be added.
        timeseries_ids (list, optional): The column names used as the timeseries IDs. Default is ["SC_variant_id", "TVKC_daydate"].
        date_id (str, optional): The column name representing the date ID. Default is "TVKC_daydate".

    Returns:
        pandas.DataFrame: The DataFrame with an additional "time_idx" column representing the time index.

    Raises:
        None.
    """
    
    # Sort the DataFrame based on the timeseries IDs
    df.sort_values(timeseries_ids, inplace=True)

    # Generate unique date list and create a dictionary with dates as keys and corresponding indices as values
    unique_dates = df[date_id].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

    # Assign time_idx using the date_to_idx dictionary
    df['time_idx'] = df[date_id].map(date_to_idx)

    # Sort the DataFrame based on the timeseries IDs and time_idx
    df.sort_values(timeseries_ids + ['time_idx'], inplace=True)

    return df

def reduce_features_with_feature_store_OneDrive(df, onedrive_api, path_feature_store):
    """
    Reduces the features of a DataFrame based on a feature store from OneDrive.

    Args:
        df (pandas.DataFrame): The DataFrame to reduce features from.
        onedrive_api (OneDriveAPI Object): Connection object for OneDrive API connection, created with class.
        path_feature_store (str): The relative path to the feature store file on OneDrive.

    Returns:
        pandas.DataFrame: The DataFrame with reduced features based on the feature store.

    Example:
        df = pd.DataFrame(...)
        path = "NOGL_shared/feature_store/fcb/feature_importances_analysis_fcb.xlsx"
        reduced_df = reduce_features_with_feature_store_OneDrive(df, path)
    """
    feature_store = onedrive_api.download_file_by_relative_path(path_feature_store) # download feature_store excel file
    feature_store = feature_store[["feature","keep"]] # 
    features_to_keep = feature_store[feature_store["keep"] == True]["feature"].to_list()
    print(f"Keeping only the following list of features for training and inference data:{features_to_keep}. This makes up only {len(features_to_keep)} number of features to previously {len(df.columns)}.")
    return df[features_to_keep]

def get_optimal_npartitions(df):
    """
    Determine the optimal value for npartitions based on DataFrame size and available system resources.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        
    Returns:
        int: The optimal value for npartitions.
    """
    cpu_cores = multiprocessing.cpu_count()
    dataframe_size = df.memory_usage().sum() / 1024 ** 2  # DataFrame size in MB
    
    # Adjust the threshold values as per your system's resources and requirements
    if dataframe_size < 1000:
        npartitions = cpu_cores  # Use the number of CPU cores
    elif dataframe_size < 10000:
        npartitions = cpu_cores * 2  # Use double the number of CPU cores
    else:
        npartitions = cpu_cores * 4  # Use four times the number of CPU cores
    print(npartitions)
    return npartitions

def reduce_mem_usage(df, verbose=False):
    """
    Reduce memory usage of a Pandas DataFrame by converting numerical columns to smaller data types.

    Parameters:
        df (DataFrame): The input DataFrame to reduce memory usage.
        verbose (bool, optional): If True, print the memory usage reduction. Defaults to False.

    Returns:
        DataFrame: The DataFrame with reduced memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    def downcast_column(col):
        col_type = col.dtypes
        if col_type in numerics:
            c_min = col.min()
            c_max = col.max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    col = col.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    col = col.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    col = col.astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    col = col.astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    col = col.astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    col = col.astype(np.float32)
                else:
                    col = col.astype(np.float64)
        return col

    npartitions = get_optimal_npartitions(df)
    df = dd.from_pandas(df, npartitions=npartitions)  # Convert DataFrame to Dask DataFrame
    df = df.map_partitions(lambda df: df.apply(downcast_column))  # Apply downcast_column function to each partition
    df = df.compute()  # Convert back to Pandas DataFrame

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

# ## 2. Load data frames

# ### 2.1 Client related downloading of tables

# create sqlalchemy connection
engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)
table_name = "transformed"

# download data

productsxdayssceleton = import_data_AWSRDS(schema=table_name,
                                           table="shopify_productsxdays_sceleton",
                                           engine=engine)
productsxdayssceleton["variant_id"] = productsxdayssceleton["variant_id"].astype("str")

shopify_sales = import_data_AWSRDS(schema=table_name,
                                   table="shopify_sales",
                                   engine=engine)
shopify_sales["variant_id"] = shopify_sales["variant_id"].astype("str")
shopify_sales = rename_columns(shopify_sales,
                               productsxdayssceleton,
                               prefix="shopify_")

klaviyo = import_data_AWSRDS(schema=table_name,
                             table="klaviyo",
                             engine=engine)
klaviyo["variant_id"] = klaviyo["variant_id"].astype("str")
klaviyo = rename_columns(klaviyo,
                         productsxdayssceleton,
                         prefix="klaviyo_")

facebook_ads = import_data_AWSRDS(schema=table_name,
                                  table="facebook_ads",
                                  engine=engine)
facebook_ads["variant_id"] = facebook_ads["variant_id"].astype("str")
facebook_ads = rename_columns(facebook_ads,
                              productsxdayssceleton,
                              prefix="facebook_ads_")

google_analytics = import_data_AWSRDS(schema=table_name,
                                      table='"google_analytics"',
                                      engine=engine)
google_analytics["variant_id"] = google_analytics["variant_id"].astype("str")
google_analytics = rename_columns(google_analytics, 
                                  productsxdayssceleton, 
                                  prefix="google_analytics_")

google_ads = import_data_AWSRDS(schema=table_name,
                                table="google_ads",
                                engine=engine)
google_ads["variant_id"] = google_ads["variant_id"].astype("str")
google_ads = rename_columns(google_ads, 
                            productsxdayssceleton, 
                            prefix="google_ads_")

marketingAndSales_plan = import_data_AWSRDS(schema=table_name,
                                            table="marketingandsales_plan",
                                            engine=engine)
marketingAndSales_plan["variant_id"] = marketingAndSales_plan["variant_id"].astype("str")
marketingAndSales_plan = rename_columns(marketingAndSales_plan, 
                                        productsxdayssceleton, 
                                        prefix="marketingAndSales_plan_")

engine.dispose()

# ### 2.2 Client non-related downloading of tables

# get config params
section = "external"
params = config(section=section,filename=filename)

# create sqlalchemy connection    
engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)
table_name = "transformed"

# download data
external_covid_data = import_data_AWSRDS(schema=table_name,
                                         table="external_covid_data",
                                         engine=engine)
external_covid_data = rename_columns(external_covid_data, 
                                     productsxdayssceleton, 
                                     prefix="external_covid_data_")

external_holidays_and_special_events_by_date = import_data_AWSRDS(schema=table_name,
                                                                  table="external_holidays_and_special_events_by_date",
                                                                  engine=engine)
external_holidays_and_special_events_by_date = rename_columns(external_holidays_and_special_events_by_date, 
                                                              productsxdayssceleton, 
                                                              prefix="external_holidays_and_special_events_by_date_")

external_weather = import_data_AWSRDS(schema=table_name,
                                      table="external_weather",
                                      engine=engine)
external_weather = rename_columns(external_weather, 
                                  productsxdayssceleton, 
                                  prefix="external_weather_")

engine.dispose()

# #### Create list of all dataframes to be merged on sceleton

all_dfs_to_be_merged = [shopify_sales,
                        klaviyo,
                        facebook_ads,
                        google_analytics,
                        google_ads,
                        marketingAndSales_plan,
                        external_covid_data,
                        external_holidays_and_special_events_by_date,
                        external_weather]

all_dfs_to_be_merged_with_daydate_and_variantid = [shopify_sales,
                        klaviyo,
                        facebook_ads,
                        google_analytics,
                        google_ads,
                        marketingAndSales_plan]

all_dfs_to_be_merged_only_daydate = [external_covid_data,
                        external_holidays_and_special_events_by_date,
                        external_weather]

all_dfs = all_dfs_to_be_merged.copy()
all_dfs.append(productsxdayssceleton)

for df in all_dfs:
    print(df.shape)

# # 3. Rename features to categorize into Time Series Data Set variables

# Remove all columns that exist in productxdayssceleton from to be merged dataframes except daydate and variant_id
for df in all_dfs_to_be_merged:
    drop_list = []
    for c in list(df.columns):
        if c in list(productsxdayssceleton.columns):
            if c not in ["daydate","variant_id"]:
                drop_list.append(c)
    if "daydate" not in list(df.columns):
        raise NameError("daydate not in columns")
    df.drop(columns=drop_list,inplace=True)

# # 4. Drop not used features

productsxdayssceleton.drop(columns=["variant_created_at", "product_published_at", "variant_max_updated_at"], inplace=True)
shopify_sales.drop(columns=["shopify_lineitems_sku","shopify_lineitems_variant_inventory_management"], inplace=True)

# ### productsxdays sceleton

productsxdayssceleton = distinguish_features(productsxdayssceleton, productsxdayssceleton,
                                     static_categoricals = ["product_category_number",
                                                            "product_category",
                                                            "product_id",
                                                            "variant_sku",
                                                            "variant_id",
                                                            "variant_inventory_item_id",
                                                            #"variant_created_at", # DROPPED
                                                            #"product_published_at", # DROPPED
                                                            "product_published_scope"],
                                     static_reals = ["variant_grams"],
                                     time_varying_known_categoricals = ["daydate", 
                                                                        "year",
                                                                        "month",
                                                                        "day",
                                                                        "weekday",
                                                                        "variant_taxable",
                                                                        "variant_requires_shipping",
                                                                        "product_status",
                                                                        "variant_inventory_management_used", 
                                                                        #"variant_max_updated_at" # DROPPED
                                                                        ],
                                     time_varying_known_reals = ["variant_RRP",
                                                                "variant_position"],
                                     time_varying_unknown_categoricals = [])


# ### shopify_sales

shopify_sales = distinguish_features(shopify_sales, productsxdayssceleton,
                                     static_categoricals = [#"shopify_lineitems_sku" # DROPPED
                                                           ],
                                     static_reals = [],
                                     time_varying_known_categoricals = [],
                                     time_varying_known_reals = ["shopify_lineitems_price",
                                                                "shopify_lineitems_variant_inventory_management",
                                                                "shopify_orders_currency_EUR",
                                                                "shopify_orders_presentment_currency_EUR",
                                                                "shopify_orders_source_web",
                                                                "shopify_rolling7days_lineitems_quantity_lastYear",
                                                                "shopify_rolling30days_lineitems_quantity_lastYear",
                                                                "shopify_rolling7days_lineitems_quantity_trend_lastYear",
                                                                "shopify_rolling7days_lineitems_quantity_season_lastYear",
                                                                "shopify_rolling7days_lineitems_quantity_resid_lastYear"],
                                     time_varying_unknown_categoricals = [#"shopify_lineitems_variant_inventory_management" # DROPPED
                                                                         ])


# ### klaviyo

klaviyo = distinguish_features(klaviyo, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = ["klaviyo_planned_recipients"],
                         time_varying_unknown_categoricals = [])


# ### facebook

facebook_ads = distinguish_features(facebook_ads, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### google_analytics

google_analytics = distinguish_features(google_analytics, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### google_ads

google_ads = distinguish_features(google_ads, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### marketingAndSales_plan

marketingAndSales_plan = distinguish_features(marketingAndSales_plan, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = ["marketingAndSales_plan_daily_FB_Budget", 
                                                     "marketingAndSales_plan_daily_GoogleLeads_Budget", 
                                                     "marketingAndSales_plan_daily_Other_Marketing_B", 
                                                     "marketingAndSales_plan_daily_Product_Sales_Target", 
                                                     "marketingAndSales_plan_daily_num_planned_klaviyo_campaigns", 
                                                     "marketingAndSales_plan_daily_planned_klaviyo_grossreach", ""],
                         time_varying_unknown_categoricals = [])


# ### external data

external_covid_data = distinguish_features(external_covid_data, productsxdayssceleton,
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])

external_weather = distinguish_features(external_weather, productsxdayssceleton,
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = ['external_weather_Berlin_humidity',
                                                     'external_weather_Berlin_rain',
                                                     'external_weather_Berlin_temp.min',
                                                     'external_weather_Berlin_temp.max',
                                                     'external_weather_Hamburg_humidity',
                                                     'external_weather_Hamburg_rain',
                                                     'external_weather_Hamburg_temp.min',
                                                     'external_weather_Hamburg_temp.max',
                                                     'external_weather_Munich_humidity',
                                                     'external_weather_Munich_rain',
                                                     'external_weather_Munich_temp.min',
                                                     'external_weather_Munich_temp.max',
                                                     'external_weather_Vienna_humidity',
                                                     'external_weather_Vienna_rain',
                                                     'external_weather_Vienna_temp.min',
                                                     'external_weather_Vienna_temp.max',
                                                     'external_weather_Zurich_humidity',
                                                     'external_weather_Zurich_rain',
                                                     'external_weather_Zurich_temp.min',
                                                     'external_weather_Zurich_temp.max'],
                         time_varying_unknown_categoricals = [])

external_holidays_and_special_events_by_date = distinguish_features(external_holidays_and_special_events_by_date, productsxdayssceleton,
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = ['external_holidays_and_special_events_by_date_daydate',
                                                           'external_holidays_and_special_events_by_date_external_importantSalesEvent',
                                                           'external_holidays_and_special_events_by_date_external_secondarySalesEvent',
                                                           'external_holidays_and_special_events_by_date_black_friday',
                                                           'external_holidays_and_special_events_by_date_cyber_monday',
                                                           'external_holidays_and_special_events_by_date_mothers_day',
                                                           'external_holidays_and_special_events_by_date_valentines_day',
                                                           'external_holidays_and_special_events_by_date_christmas_eve',
                                                           'external_holidays_and_special_events_by_date_fathers_day',
                                                           'external_holidays_and_special_events_by_date_orthodox_new_year',
                                                           'external_holidays_and_special_events_by_date_chinese_new_year',
                                                           'external_holidays_and_special_events_by_date_rosenmontag',
                                                           'external_holidays_and_special_events_by_date_carneval',
                                                           'external_holidays_and_special_events_by_date_start_of_ramadan',
                                                           'external_holidays_and_special_events_by_date_start_of_eurovision',
                                                           'external_holidays_and_special_events_by_date_halloween',
                                                           'external_holidays_and_special_events_by_date_saint_nicholas',
                                                           'external_holidays_and_special_events_by_date_external_holiday'],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# # 5. Adjust datatypes

# transform based on variable type
for df in all_dfs:
    for c in df.columns:
        if (c.startswith("SC_") or c.startswith("TVKC_") or c.startswith("TVUC_")):
            df[c] = df[c].astype("category")
        if (c.startswith("SR_") or c.startswith("TVKR_") or c.startswith("TVUR_")):
            if all(df[c] % 1 == 0):  # Check if all values in the column are whole numbers
                df[c] = df[c].astype("int")
            else:
                df[c] = df[c].astype("float")

for df in all_dfs:
    df["TVKC_daydate"] = pd.to_datetime(df["TVKC_daydate"])

# # 6. Merge

# #### Merge all scelecton based dataframes

df_consolidated = merge_with_productsxdaysscelecton(productsxdayssceleton, all_dfs_to_be_merged_with_daydate_and_variantid, ["TVKC_daydate","SC_variant_id"])
df_consolidated = df_consolidated.drop_duplicates()

print("Merged all sceleton based dataframes with daydate and variant_id. Shape is now: ", df_consolidated.shape)

# Free up memory
for df in all_dfs_to_be_merged_with_daydate_and_variantid:
    del df
del all_dfs_to_be_merged_with_daydate_and_variantid

# Call the garbage collector
gc.collect()

# #### Merge all only daydate based dataframes

df_consolidated = merge_with_productsxdaysscelecton(df_consolidated, all_dfs_to_be_merged_only_daydate, ["TVKC_daydate"])
df_consolidated = df_consolidated.drop_duplicates()

print("Merged all daydate only dataframes with daydate only as well. Shape is now: ", df_consolidated.shape)

# # 7. Temporal feature engineering

# ### Split in day, month, year and weekday

df_consolidated["year"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%Y')))
df_consolidated["month"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%m')))
df_consolidated["day"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%d')))
df_consolidated["weekday"] = df_consolidated['TVKC_daydate'].apply(lambda row: pd.to_datetime(row).isoweekday())

# ### add time_idx

df_consolidated = add_time_idx(df_consolidated, timeseries_ids = ["SC_variant_id","TVKC_daydate"], date_id = "TVKC_daydate")

# ### add variable categories to new features

df_consolidated.rename(columns={"year":"TVKC_year",
                                "month":"TVKC_month",
                                "day":"TVKC_day",
                                "weekday":"TVKC_weekday",
                                "time_idx":"TVKR_time_idx"}, inplace=True)

# ### change dtype regarding variable categories of new features

for c in ["TVKC_year","TVKC_month","TVKC_day","TVKC_weekday"]:
    df_consolidated[c] = df_consolidated[c].astype("category")
    
df_consolidated["TVKR_time_idx"] = df_consolidated["TVKR_time_idx"].astype("int")

# # 8. Final checks and corrections

# ### check for nans
print("NaN values in df:", df_consolidated.isna().sum().sum())

# ### "." to "_" in feature names
print(". to _ in feature names starting")
for c in df_consolidated.columns:
    if "." in c:
        print("Renaming:", c, " to:", c.replace(".","_"))
        df_consolidated.rename(columns={c:c.replace(".","_")},inplace=True)
print(". to _ in feature names finished")

# replace inf values

df_consolidated.replace([np.inf, -np.inf], 0, inplace=True)

# # 9. Analysis of cutoff timings and data check per source

# run analysis
df_results, df_results_min_analysis, df_results_max_analysis = get_minmax_above0_on_feature_level(df_consolidated, "TVKC_daydate")

# save results to DB
section = args.client_name # section = client_name
params = config(section=section,filename=filename)

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)

df_results.to_sql("data_analysis_overview", con = engine, schema="xai", if_exists='replace', index=False, chunksize=1000, method="multi")
df_results_min_analysis.to_sql("data_analysis_minimum_timepoints", con = engine, schema="xai", if_exists='replace', index=False, chunksize=1000, method="multi")
df_results_max_analysis.to_sql("data_analysis_maximum_timepoints", con = engine, schema="xai", if_exists='replace', index=False, chunksize=1000, method="multi")

engine.dispose()

# # 10. Reduce number of features with OneDrive Feature Store
print("Feature reduction started.")
onedrive_api = OneDriveAPI(client_id, client_secret, tenant_id, msal_scope, site_id)
df_consolidated = reduce_features_with_feature_store_OneDrive(df_consolidated, onedrive_api, "NOGL_shared/feature_store/feature_importances_analysis.xlsx")
print("Feature reduction finished.")

# # 11. Categorization of products for different TFT-trainings

# set parameters for filtering
number_of_historic_days = 380

# for TFT with normal sales quantty
number_of_sales_top = number_of_historic_days*2 #100

# for TFT with 7daysaverage
number_of_sales_buttom = number_of_historic_days/2 #50

# only for category analysis
year_to_filter_by = 2022

# filter by category and year to split into categories
categorization = df_consolidated.groupby(["SC_product_category","TVKC_year"], as_index=False).agg({"TVUR_shopify_lineitems_quantity":"sum",
                                                                                        "SC_variant_id":"nunique",
                                                                                        "SC_product_id":"nunique"})

# calculate category ratio
categorization["ratio"] = categorization["TVUR_shopify_lineitems_quantity"]/categorization["SC_variant_id"]

categorization = categorization[categorization["TVKC_year"] == year_to_filter_by]
categorization = categorization.sort_values(by="ratio", ascending=False)

YoY = datetime.now() - timedelta(days=number_of_historic_days)

categorization_variant_id = df_consolidated[pd.to_datetime(df_consolidated["TVKC_daydate"]) > YoY]

categorization_variant_id = categorization_variant_id.groupby(["SC_product_category","SC_variant_id"], as_index=False)["TVUR_shopify_lineitems_quantity"].sum()

categorization_variant_id = categorization_variant_id.sort_values(by="TVUR_shopify_lineitems_quantity", ascending=False)

categorization_variant_id_top = categorization_variant_id[categorization_variant_id["TVUR_shopify_lineitems_quantity"] >= number_of_sales_top]
categorization_variant_id_buttom = categorization_variant_id[(categorization_variant_id["TVUR_shopify_lineitems_quantity"] < number_of_sales_top) &
                                                            (categorization_variant_id["TVUR_shopify_lineitems_quantity"] >= number_of_sales_buttom)]
categorization_variant_id_kicked = categorization_variant_id[categorization_variant_id["TVUR_shopify_lineitems_quantity"] < number_of_sales_buttom]

print("Number of top products above", number_of_sales_top, "in sales:", len(categorization_variant_id_top))
print("Number of 7daysaverage products between", number_of_sales_top,"and",number_of_sales_buttom, "in sales:", len(categorization_variant_id_buttom))
print("Number of to be kicked products below", number_of_sales_buttom, "in sales:", len(categorization_variant_id_kicked))

data_top = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_top.SC_variant_id))]
data_buttom = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_buttom.SC_variant_id))]
data_kicked = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_kicked.SC_variant_id))]

# # 12. Memory usage reduction before export
print("Memory reduction started.")
data_top = reduce_mem_usage(data_top, verbose=True)
data_buttom = reduce_mem_usage(data_buttom, verbose=True)
data_kicked = reduce_mem_usage(data_kicked, verbose=True)
print("Memory reduction finished.")

print("After memory reduction SC_variant_id in top seller data is of type:", data_top["SC_variant_id"].dtype)

# # 13. Export

# ## Split top, buttom (long-tail) and kicked products

data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated.csv")
data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated.csv")
data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated.csv")
pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated.csv")

data_top = cutoff_beginning_of_timeseries(data_top, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")

data_buttom = cutoff_beginning_of_timeseries(data_buttom, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")

data_kicked = cutoff_beginning_of_timeseries(data_kicked, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")

data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated_cutoff.csv")
data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated_cutoff.csv")
data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated_cutoff.csv")
pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated_cutoff.csv")

# ## Whole dataset

df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated.csv")

df_consolidated = cutoff_beginning_of_timeseries(df_consolidated, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")

df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated_cutoff.csv")

