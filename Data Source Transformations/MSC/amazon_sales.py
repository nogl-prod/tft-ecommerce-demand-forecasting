#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 1. Imports/Options

# ## 1.1 External imports

import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine    
import re
import multiprocessing
import dask.dataframe as dd

# ## 1.2 Internal imports

import sys
import os
# UNCOMMEND LATER & REPLACE
# prefix = '/opt/ml'
# src  = os.path.join(prefix, 'processing/input/')
# sys.path.append(src)

# import the 'config' funtion from the config.py file

from configAWSRDS import config

# from support functions:
from utils import *
from Support_Functions import *
from static_variables import datacollection_startdate

# UNCOMMEND LATER
# import argparse
# # Create the parser
# parser = argparse.ArgumentParser()
# # Add argument
# parser.add_argument("--client_name", type=str, required=True)
# # Parse the argument
# args = parser.parse_args()


# # get config params
# section = args.client_name
# print("client name:", section)
# filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
# params = config(section=section,filename=filename)

# create sqlalchemy connection 

client_name = "dogs-n-tiger"


channel_names = get_channel_name(client_name)
engines = {
    "dev" : get_db_connection(dbtype="dev", client_name=client_name),
    "prod" : get_db_connection(dbtype="prod", client_name=client_name)
}
# ## 1.3 Options
engine = engines["dev"]

pd.set_option('display.max_columns', None)


# In[2]:


# Support functions

def clean_data(data, fill_values):
    """
    Cleans the data by filling missing values based on the specified fill_values dictionary.

    :param data: DataFrame containing the data to be cleaned
    :param fill_values: Dictionary mapping columns to fill values
    :return: Cleaned DataFrame
    """
    # Making a copy of the data to avoid modifying the original
    cleaned_data = data.copy()
    
    # Iterating through the fill_values dictionary and filling missing values
    for column, fill_value in fill_values.items():
        # Checking if the column exists in the data
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].fillna(fill_value)

    return cleaned_data

def replace_none_with_nan(data, columns_to_replace=None):
    """
    Replaces the string value "None" with np.nan in the specified columns.

    :param data: DataFrame containing the data
    :param columns_to_replace: List of column names where the replacement should be applied (None for all columns)
    :return: DataFrame with replaced values
    """
    # Making a copy of the data to avoid modifying the original
    replaced_data = data.copy()

    # If no specific columns are provided, apply the replacement to all columns
    if columns_to_replace is None:
        columns_to_replace = replaced_data.columns

    # Applying the replacement in the specified columns
    for column in columns_to_replace:
        if column in replaced_data.columns:
            replaced_data[column] = replaced_data[column].replace('None', np.nan)
            replaced_data[column] = replaced_data[column].fillna(np.nan)

    return replaced_data

def replace_nan_with_zeros(data, columns_to_replace=None):
    """
    Replaces NaN values with 0 values in the specified columns.

    :param data: DataFrame containing the data
    :param columns_to_replace: List of column names where the replacement should be applied (None for all columns)
    :return: DataFrame with replaced values
    """

    # Making a copy of the data to avoid modifying the original
    replaced_data = data.copy()

    # If no specific columns are provided, apply the replacement to all columns
    if columns_to_replace is None:
        columns_to_replace = replaced_data.columns

    # Applying the replacement in the specified columns
    for column in columns_to_replace:
        if column in replaced_data.columns:
            replaced_data[column] = replaced_data[column].replace(np.nan, 0)

    return replaced_data

def compute_average_delivery_time(df):
    """
    This function calculates the average delivery date and delivery time for each order in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame. It should include the following columns:
        'order_earliest_delivery_date', 'order_latest_delivery_date', 'order_purchase_date'
        
    Returns:
    df (pandas.DataFrame): The input DataFrame with two new columns added: 
        'average_delivery_date' and 'delivery_time'. 
        'average_delivery_date' is the midpoint between the earliest and latest delivery dates. 
        'delivery_time' is the number of days (as a decimal) between the purchase date and the average delivery date.
    """

    import pandas as pd
    
    # Parse the necessary columns into datetime format
    columns_to_parse = ['order_earliest_delivery_date', 'order_latest_delivery_date', 'order_purchase_date']

    for column in columns_to_parse:
        df[column] = pd.to_datetime(df[column], utc=True, errors='coerce')

    # Compute the average delivery time
    df['average_delivery_date'] = df['order_earliest_delivery_date'] + \
                                  (df['order_latest_delivery_date'] - df['order_earliest_delivery_date']) / 2

    # Compute the time from order purchase to average delivery date in days as a decimal
    df['delivery_time_to_consumer'] = round((df['average_delivery_date'] - df['order_purchase_date']).dt.total_seconds() / 86400, 2)

    df.drop(columns=["average_delivery_date",
                     "order_earliest_delivery_date",
                     "order_latest_delivery_date"],
                     inplace=True)
    
    return df

def make_true(df, column, value):
    """
    This function will set a specific value to True and the rest to False. 

    Parameters:
    df (dataframe): The dataframe that contains the column that needs to be transformed
    column (string): The column that needs to be transformed
    value (string): The value that needs to be set to True
    """
    df[column] = df[column].replace(value, True)
    for i in df[column].unique():
        if i != True:
            df[column] = df[column].replace(i, False)
    return df

def transform_to_boolean(df, subset_columns):
    """
    This function attempts to transform a subset of columns in a DataFrame to boolean type.
    If a column can be converted to boolean, it does so and fills NaN values with False.
    If a column can't be converted to boolean, it sets NaN, False, and None values to False, 
    and all other values to True.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame whose columns are to be transformed.
    subset_columns (list): A list of column names to transform.
    
    Returns:
    df (pandas.DataFrame): The DataFrame with the transformed columns.
    """
    for column in subset_columns:
        df[column] = df[column].fillna(np.nan)
        # check if column datatype is boolean
        if df[column].dtype == bool:
            print(f"{column} is already boolean")
            continue
        else:
            # If conversion to boolean fails, set NaN, False, and None values to False, and all other values to True
            print(f"{column} is not yet a boolean. Setting NaN, False, and None values to False, and all other values to True.")
            df[column] = ~df[column].isin([np.nan, False, None])
        
        # Fill NaN values with False
        df[column] = df[column].fillna(False)
    
    return df

def one_hot_encode(df, columns):
    """
    One hot encodes the columns in the list columns and drops the original columns
    
    Parameters:
    df (DataFrame): The DataFrame to be transformed
    columns (list): The list of columns to be one hot encoded
    """
    for c in columns:
        df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)
        df.drop(c, axis=1, inplace=True)
    return df

def clean_concatenated_values(value):
    """
    Cleans a value which may be a concatenated string of repeated numbers.

    The function will attempt to convert the value to a float. If the conversion fails,
    it will check if the value is made up of repeated patterns (e.g., "3.003.003.00").
    If such a pattern is detected, the function returns the first repeated value as a float.
    If not, it computes the mean of all numeric parts found in the string.

    Parameters:
    - value (str): The string value to be cleaned.

    Returns:
    - float: The cleaned float value or the computed mean of numeric parts in the string.
    """
    try:
        # Try to convert value to float
        return float(value)
    except ValueError:
        # If conversion fails, look for repeated patterns
        repeated_pattern_match = re.match(r'((\d+\.\d+)(?:\2)+)', value)
        
        if repeated_pattern_match:
            return float(repeated_pattern_match.group(2))
        
        # Otherwise, compute the mean of the numeric parts
        numeric_parts = re.findall(r"\d+\.\d+", value)
        numeric_parts = [float(part) for part in numeric_parts]
        return sum(numeric_parts) / len(numeric_parts) if numeric_parts else value

def transform_dtypes(df, dtypes_dict, datacollection_startdate=np.nan):
    """
    Transforms the data types of the given DataFrame's columns based on the provided dtypes_dict.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns' data types need to be transformed.
    - dtypes_dict (dict): A dictionary containing the desired data types for each column.
                          The keys should be column names, and the values should be the desired data types.
                          Example: {'column1': 'int64', 'column2': 'str'}
    - datacollection_startdate (optional, default=np.nan): A value used to fill NaN values for 'datetime' dtype.

    Returns:
    - pd.DataFrame: The DataFrame with the transformed data types.
    """
    def find_fillna_value(dtype):
        if (dtype == "int64") or (dtype == "float64"):
            return 0
        if dtype == "str":
            return ""
        if dtype == "boolean":
            return False
        if dtype == "datetime":
            return datacollection_startdate
        if dtype == "category":
            return ""

    # Filter out the unwanted keys from dtypes_dict that don't match the columns in the DataFrame
    dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in df.columns}

    transformation_df = pd.DataFrame.from_dict(dtypes_dict, orient="index").reset_index().rename(columns={0:"dtype","index":"feature"})
    transformation_df["fillna_value"] = transformation_df["dtype"].apply(lambda x: find_fillna_value(x))
    transformation_df = transformation_df.merge(pd.DataFrame(df.isna().sum()).reset_index().rename(columns={0:"number_of_NaN_values","index":"feature"}), how="left", on="feature")

    # dict_dtypechanges = pd.Series(transformation_df.loc[(transformation_df["dtype"] != "datetime")].dtype.values, index=transformation_df.loc[(transformation_df["dtype"] != "datetime")].feature).to_dict()
    # df = df.astype(dict_dtypechanges)

    # Separate datetime columns and handle them using pd.to_datetime
    datetime_columns = transformation_df.loc[transformation_df["dtype"] == "datetime"].feature
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

    # Handle other data types using astype
    non_datetime_columns = transformation_df.loc[transformation_df["dtype"] != "datetime"]
    dict_dtypechanges = pd.Series(non_datetime_columns.dtype.values, index=non_datetime_columns.feature).to_dict()
    df = df.astype(dict_dtypechanges)

    return df

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

def STL_perProduct(df, identifiers, feature_toSTL):
    # Create a tuple of identifiers to deal with multi-level groupings
    unique_identifiers = df[identifiers].drop_duplicates().to_records(index=False).tolist()
    
    STL_results = pd.DataFrame(columns=identifiers + ["daydate"])

    for identifier_values in unique_identifiers:
        mask = (df.set_index(identifiers).index == identifier_values)
        sub_df = df[mask]
        
        res = STL(sub_df.set_index("daydate")[feature_toSTL], period=30).fit()
        plevel_results = pd.DataFrame(res.trend, columns=[feature_toSTL + "_trend"])
        plevel_results[feature_toSTL + "_season"] = res.seasonal
        plevel_results[feature_toSTL + "_resid"] = res.resid
        
        plevel_results.reset_index(inplace=True)
        
        # Assign the identifier values back to the results
        for identifier, value in zip(identifiers, identifier_values):
            plevel_results[identifier] = value
            
        STL_results = pd.concat([STL_results, plevel_results], ignore_index=True)

    # Merge using all identifiers
    df = df.merge(STL_results, how="left", on=identifiers + ["daydate"])
    df.drop_duplicates(inplace=True)

    return df

def lag_by_oneYear(df, column_name_to_lag, identifiers):
    df = df.copy()
    df_newData = df.copy()
    df_newData["daydate_plus1y"] = df_newData["daydate"] + relativedelta(months=12)
    
    # Include all identifiers in the subset
    columns_to_select = identifiers + ["daydate_plus1y", column_name_to_lag]
    df_newData = df_newData[columns_to_select]
    
    # Group by all identifiers
    groupby_columns = ["daydate_plus1y"] + identifiers
    df_newData = df_newData.groupby(groupby_columns, as_index=False).mean()
    
    # Rename column
    df_newData.rename(columns={column_name_to_lag: column_name_to_lag + "_lastYear"}, inplace=True)
    
    # Merge on all identifiers
    merge_on_columns = ["daydate"] + identifiers
    df = df.merge(df_newData, how="left", left_on=merge_on_columns, right_on=["daydate_plus1y"] + identifiers)
    
    df.drop(columns="daydate_plus1y", inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Sort by all identifiers
    sort_columns = ["daydate"] + identifiers
    df.sort_values(by=sort_columns, inplace=True)
    
    # Fill NaNs
    df[column_name_to_lag + "_lastYear"] = df[column_name_to_lag + "_lastYear"].fillna(method="ffill")
    
    return df


# In[3]:


amazon_sales = import_data_AWSRDS(schema="amazon",table="order_line_items",engine=engine)
print(amazon_sales.shape)

amazon_sceleton = import_data_AWSRDS(schema="sc_amazon",table="amazon_productsxdays_sceleton",engine=engine)
print(amazon_sceleton.shape)


# In[4]:


# selected only the columns that are needed for the amazon_sales transformation
# Creating the list of column names for each category

hierarchy = ['lineitem_asin',
             'lineitem_seller_sku',
             'order_marketplace_id' # Include this as hierarchy for timeseries identification
             ]

dates = ['order_purchase_date']

keep_as_is = ['lineitem_quantity_ordered',
    'order_order_status',
    'lineitem_shippingprice_amount',
    'lineitem_shippingtax_amount',
    'lineitem_shippingdiscount_amount',
    'lineitem_shippingdiscounttax_amount',
    'lineitem_item_price_amount',
    'lineitem_item_tax_amount',
    'lineitem_promotion_discount_amount',
    'lineitem_promotion_discount_tax_amount',
    'lineitem_buyer_info_giftwrap_price_amount',
    'lineitem_buyer_info_giftwrap_tax_amount'
]

feature_engineering = ['order_earliest_delivery_date', # Engineer feature that can check for average delivery time
                       'order_latest_delivery_date']

boolean = [
    'order_is_prime',
    'order_is_premium_order',
    'order_is_global_express_enabled',
    'lineitem_is_gift',
    'lineitem_promotion_ids', # Probably unnecessary, because lineitems_promotion_discount_amount is already included
    'order_automatedshipsettings_hasautomatedshipsettings',
    'lineitem_condition_id',
    'order_is_business_order',
    'order_fulfillment_channel', # Make this is boolean checking for AFN (Amazon Fullfillment Network) vs. MFN (Merchant Fullfillment Network) with AFN being True
    'order_shipment_service_level_category', # Make this is boolean checking for Expedited vs. Standard with Expedited being True
    'order_order_total_currency_code', # Make this is boolean checking for EUR vs. other currencies with EUR being True
    "order_shipping_address_country_code", # Make this is boolean checking for DE vs. other countries with DE being True
]

# one_hot_encoding = [
#     'order_payment_method']


# Combining all the lists into one
columns_to_keep = hierarchy + dates + keep_as_is + feature_engineering + boolean #+ one_hot_encoding


# In[5]:


# create basis for cleaned amazon_sales df

# get rid of duplicates
print("Before duplicates removal: ", amazon_sales.shape)
amazon_sales.sort_values(by=["order_last_update_date", "lineitem_quantity_shipped"], ascending=[False, False], inplace=True)
amazon_sales = amazon_sales.drop_duplicates(subset=(hierarchy + dates + ["order_amazon_order_id"]), keep='first')
print("After duplicates removal: ", amazon_sales.shape)

amazon_sales_clean = amazon_sales[columns_to_keep].copy()


# In[6]:


# clean before aggregation

# fill None and NaN values
amazon_sales_clean = replace_none_with_nan(amazon_sales_clean)
amazon_sales_clean = replace_nan_with_zeros(amazon_sales_clean, keep_as_is)


# In[7]:


# transform before aggregation

# feature engineering
amazon_sales_clean = compute_average_delivery_time(amazon_sales_clean)

# date
for c in dates:
    amazon_sales_clean[c] = pd.to_datetime(amazon_sales_clean[c], utc=True, errors='coerce').dt.strftime("%Y-%m-%d")

# boolean
amazon_sales_clean = make_true(amazon_sales_clean, "lineitem_is_gift", "true") # make "true" True the rest true
amazon_sales_clean = make_true(amazon_sales_clean, "lineitem_condition_id", "New") # make "New" True the rest false
amazon_sales_clean = make_true(amazon_sales_clean, "order_fulfillment_channel", "AFN") # make AFN True the rest false
amazon_sales_clean = make_true(amazon_sales_clean, "order_shipment_service_level_category", "Expedited") # make Expedited True the rest false
amazon_sales_clean = make_true(amazon_sales_clean, "order_order_total_currency_code", "EUR") # make EUR True the rest false
amazon_sales_clean = make_true(amazon_sales_clean, "order_shipping_address_country_code", "DE") # make DE True the rest false
amazon_sales_clean["lineitem_promotion_ids"] = amazon_sales_clean["lineitem_promotion_ids"].replace("None", np.nan)
amazon_sales_clean = transform_to_boolean(amazon_sales_clean, boolean)
amazon_sales_clean.rename(columns={"lineitem_promotion_ids": "lineitem_has_promotion_ids",
                                   "lineitem_condition_id": "lineitem_condition_is_new",
                                   "order_fulfillment_channel": "fullfillment_channel_is_AFN",
                                   "order_shipment_service_level_category": "shipment_servicelevel_is_expedited",
                                   "order_order_total_currency_code": "order_currency_is_EUR",
                                   "order_shipping_address_country_code": "shipping_address_is_DE"}, 
                                   inplace=True)


# one hot encoding
# amazon_sales_clean = one_hot_encode(amazon_sales_clean, one_hot_encoding) -> Currently not used, because the number of payment methods is not fixed, so data structure could vary

# only keep the not cancelled orders
amazon_sales_clean = amazon_sales_clean[amazon_sales_clean["order_order_status"] != "Canceled"] # Pending is still inlcuded, because it is the responsibility of the seller to integrate this information in the inventory data
amazon_sales_clean.drop(columns=["order_order_status"], inplace=True)

# final fillna
amazon_sales_clean.fillna(0, inplace=True)


# In[8]:


# groupby / aggregate sales by the date order_purchase_date?
aggregation_dict = {
    'lineitem_quantity_ordered': 'sum',
    'lineitem_shippingprice_amount': 'mean',
    'lineitem_shippingtax_amount': 'mean',
    'lineitem_shippingdiscount_amount': 'mean',
    'lineitem_shippingdiscounttax_amount': 'mean',
    'lineitem_item_price_amount': 'mean',
    'lineitem_item_tax_amount': 'mean',
    'lineitem_promotion_discount_amount': 'mean',
    'lineitem_promotion_discount_tax_amount': 'mean',
    'lineitem_buyer_info_giftwrap_price_amount': 'mean',
    'lineitem_buyer_info_giftwrap_tax_amount': 'mean',
    'order_is_prime': 'mean',
    'order_is_premium_order': 'mean',
    'order_is_global_express_enabled': 'mean',
    'lineitem_is_gift': 'mean',
    'lineitem_has_promotion_ids': 'mean',
    'order_automatedshipsettings_hasautomatedshipsettings': 'mean',
    'lineitem_condition_is_new': 'mean',
    'order_is_business_order': 'mean',
    'fullfillment_channel_is_AFN': 'mean',
    'shipment_servicelevel_is_expedited': 'mean',
    'order_currency_is_EUR': 'mean',
    'shipping_address_is_DE': 'mean',
    'delivery_time_to_consumer': 'mean'
}

# adjust dtype for accurate aggregation
for k in aggregation_dict.keys():
    amazon_sales_clean[k].astype("float64")

# clean concatenated values
for k in aggregation_dict.keys():
    amazon_sales_clean[k] = amazon_sales_clean[k].apply(clean_concatenated_values)

# group all "to group" columns with respective aggregation functions
amazon_sales_aggregated = amazon_sales_clean.groupby(["order_marketplace_id",
                                                            "lineitem_asin",
                                                            "lineitem_seller_sku",
                                                            "order_purchase_date"],
                                                            as_index=False).agg(aggregation_dict)


# In[9]:


# dtype transformation and reduction

# Creating dtype mapping dictionary
dtypes_dict = {col: 'str' if col in ['order_marketplace_id', 'lineitem_asin', 'lineitem_seller_sku'] else 
               'datetime' if col in ['order_purchase_date'] else 
               'int64' if col in ['lineitem_quantity_ordered_sum'] else
               'float64' for col in amazon_sales_aggregated.columns}

# Transforming dtypes
amazon_sales_aggregated = transform_dtypes(amazon_sales_aggregated, dtypes_dict, datacollection_startdate=np.nan)

# Reducing dtypes to save memory
amazon_sales_aggregated = reduce_mem_usage(amazon_sales_aggregated, verbose=True)


# In[10]:


# rename all columns to match data structure
rename_dict = {'order_marketplace_id': 'marketplace_id',
                'lineitem_asin': 'variant_asin',
                'lineitem_seller_sku': 'variant_sku',
                'order_purchase_date': 'daydate',
                'lineitem_quantity_ordered': 'lineitems_quantity',
                'lineitem_item_price_amount': 'lineitems_price',
                'lineitem_item_tax_amount': 'lineitems_tax',
                'lineitem_promotion_discount_amount': 'lineitems_discountallocations_amount',
                'lineitem_promotion_discount_tax_amount': 'lineitems_discountallocations_amount_tax',
                'lineitem_shippingprice_amount': 'lineitems_shippingprice',
                'lineitem_shippingtax_amount': 'lineitems_shippingtax',
                'lineitem_shippingdiscount_amount': 'lineitems_shippingdiscount',
                'lineitem_shippingdiscounttax_amount': 'lineitems_shippingdiscount_tax',
                'lineitem_buyer_info_giftwrap_price_amount': 'lineitems_giftwrap_price',
                'lineitem_buyer_info_giftwrap_tax_amount': 'lineitems_giftwrap_tax',
                'order_is_prim': 'order_is_prime',
                'order_is_premium_order': 'order_is_premium_order',
                'order_is_global_express_enabled': 'order_is_global_express_enabled',
                'lineitem_is_gift': 'lineitem_is_gift',
                'lineitem_has_promotion_ids': 'lineitem_has_promotion_ids',
                'order_automatedshipsettings_hasautomatedshipsettings': 'order_automatedshipsettings_hasautomatedshipsettings',
                'lineitem_condition_is_new': 'lineitem_condition_is_new',
                'order_is_business_order': 'order_is_business_order',
                'fullfillment_channel_is_AFN': 'fullfillment_channel_is_AFN',
                'shipment_servicelevel_is_expedited': 'shipment_servicelevel_is_expedited',
                'order_currency_is_EUR': 'order_currency_is_EUR',
                'shipping_address_is_DE': 'shipping_address_is_DE',
                'delivery_time_to_consumer': 'delivery_time_to_consumer'}

amazon_sales_aggregated = amazon_sales_aggregated.rename(columns=rename_dict)


# In[11]:


amazon_sceleton.columns


# In[12]:


# merge on sceleton
amazon_sceleton.columns = amazon_sceleton.columns.str.replace("amazon_", "")
# rename nogl_id to variant_id
amazon_sceleton.rename(columns={"nogl_id": "variant_id"}, inplace=True)
# change dtype of shopify_productxdays_sceleton to merge
amazon_sceleton["daydate"] = pd.to_datetime(amazon_sceleton["daydate"], utc=True).dt.date
amazon_sales_aggregated["daydate"] = pd.to_datetime(amazon_sales_aggregated["daydate"], utc=True).dt.date

# merge
amazon_sales_final = amazon_sceleton.merge(amazon_sales_aggregated.drop(columns=["marketplace_id"]), how="left", on=["variant_asin",
                                                                                                                    "variant_sku",
                                                                                                                    "daydate"])


# In[13]:


# fillnan values

# forward fill na values for selected columns

# Set cost-related columns to 0 where lineitems_quantity is 0 or NaN
cost_related_columns = [
    'lineitems_shippingdiscount',
    "lineitems_shippingdiscount_tax",
    'lineitems_discountallocations_amount', 
    'lineitems_discountallocations_amount_tax', 
    'lineitems_giftwrap_price', 
    'lineitems_giftwrap_tax'
]

# Set boolean/flag columns to 0 where lineitems_quantity is 0 or NaN
flag_columns = [
    'order_is_prime', 
    'order_is_premium_order', 
    'lineitem_is_gift', 
    'lineitem_has_promotion_ids', 
    'order_automatedshipsettings_hasautomatedshipsettings', 
    'lineitem_condition_is_new', 
    'order_is_business_order', 
    'fullfillment_channel_is_AFN', 
    'shipment_servicelevel_is_expedited',
    'shipping_address_is_DE', 
]

# Forward fill the columns that are static or change slowly
static_columns = [
    'order_currency_is_EUR', 
    'delivery_time_to_consumer', 
    'lineitems_shippingprice',
    'lineitems_shippingtax',
    'lineitems_price',
    'lineitems_tax',
    'order_is_global_express_enabled'
]

# Convert to float datatype
for c in static_columns:
    amazon_sales_final[c] = amazon_sales_final[c].astype("float64")

# Sort by unique time series
amazon_sales_final.sort_values(by=["variant_asin", "variant_sku", "daydate"], inplace=True)

# Group by the unique time series and then forward fill within each group
for col in static_columns:
    amazon_sales_final[col] = amazon_sales_final.groupby(["variant_asin", "variant_sku"])[col].transform(lambda x: x.fillna(method='ffill'))

# Fill any remaining NaN values with 0
amazon_sales_final.fillna(0, inplace=True)


# In[14]:


amazon_sales_final_engineered = amazon_sales_final.copy()


# In[15]:


# feature engineering from shopify_sales

import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import STL

# Assuming your Amazon sales data is named 'amazon_sales'

# Rolling Averages and Differences
# Sort dataframe
amazon_sales_final_engineered = amazon_sales_final_engineered.sort_values(["variant_asin", "variant_sku", "daydate"]).reset_index(drop=True)

# Specify which features to transform
features_to_avg = ["lineitems_quantity"]

# Rolling average last 7 days
for i in features_to_avg:
    amazon_sales_final_engineered["rolling7days_"+i] = amazon_sales_final_engineered.groupby(by=["variant_asin", "variant_sku"])[i].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
# Rolling average last 30 days
for i in features_to_avg:
    amazon_sales_final_engineered["rolling30days_"+i] = amazon_sales_final_engineered.groupby(by=["variant_asin", "variant_sku"])[i].transform(lambda x: x.rolling(30, min_periods=1).mean())

# Difference to rolling
for i in features_to_avg:
    amazon_sales_final_engineered["delta7days_"+i] = amazon_sales_final_engineered[i] - amazon_sales_final_engineered["rolling7days_"+i]
    amazon_sales_final_engineered["delta30days_"+i] = amazon_sales_final_engineered[i] - amazon_sales_final_engineered["rolling30days_"+i]

# Fill last years' averages in this year's data
for i in features_to_avg:
    amazon_sales_final_engineered = lag_by_oneYear(amazon_sales_final_engineered, "rolling7days_"+i, ["variant_asin", "variant_sku"])
    amazon_sales_final_engineered = lag_by_oneYear(amazon_sales_final_engineered, "rolling30days_"+i, ["variant_asin", "variant_sku"])
    
amazon_sales_final_engineered.fillna(0, inplace=True)

# Logs
for i in features_to_avg:
    amazon_sales_final_engineered["log_"+i] = np.log(amazon_sales_final_engineered[i] + 1e-8)

# STL decomposition

# Apply STL for 7 days rolling
amazon_sales_final_engineered = amazon_sales_final_engineered.sort_values(by=["variant_asin", "variant_sku","daydate"])
amazon_sales_final_engineered = STL_perProduct(amazon_sales_final_engineered, ["variant_asin", "variant_sku"], "rolling7days_lineitems_quantity")
amazon_sales_final_engineered.fillna(0, inplace=True)

# Fill +1 year with previous year trend, season, and resid
amazon_sales_final_engineered = amazon_sales_final_engineered.sort_values(by=["variant_asin", "variant_sku","daydate"])
for i in ["trend","season","resid"]:
    amazon_sales_final_engineered = lag_by_oneYear(amazon_sales_final_engineered, ("rolling7days_lineitems_quantity_"+i), ["variant_asin", "variant_sku"])
    amazon_sales_final_engineered.fillna(0, inplace=True)

amazon_sales_final_engineered.reset_index(drop=True, inplace=True)

# get rid of negative or positive inifitiy values in log_lineitems_quantity
amazon_sales_final_engineered.loc[amazon_sales_final_engineered.log_lineitems_quantity == -np.inf, 'log_lineitems_quantity'] = 0
amazon_sales_final_engineered.loc[amazon_sales_final_engineered.log_lineitems_quantity == np.inf, 'log_lineitems_quantity'] = 0

amazon_sales_final_engineered


# In[17]:


# dtype transformation and reduction

# Creating dtype mapping dictionary
dtypes_dict = {col: 'str' if col in ['product_category', 'product_id', 'variant_sku', 'variant_barcode', 'variant_asin', 'product_status', 'product_condition', 'variant_id'] else 
               'datetime' if col in ['daydate'] else 
               'int64' if col in ['lineitems_quantity'] else
               'float64' for col in amazon_sales_final_engineered.columns}

# Transforming dtypes
amazon_sales_final_engineered = transform_dtypes(amazon_sales_final_engineered, dtypes_dict, datacollection_startdate=np.nan)

# Reducing dtypes to save memory
amazon_sales_final_engineered = reduce_mem_usage(amazon_sales_final_engineered, verbose=True)


# In[18]:


print("Amazon sales dataframe shape:", amazon_sales_final_engineered.shape)
print(amazon_sales_final_engineered.info())


# In[ ]:


# upload to database
t = Timer("Export")
amazon_sales_final_engineered.to_sql("msc_amazon_sales", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
amazon_sales_final_engineered.to_sql("historical_sales", con = engine, schema="sc_amazon", if_exists='replace', index=False, chunksize=1000, method="multi")

t.end()

