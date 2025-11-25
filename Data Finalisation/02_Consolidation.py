#!/usr/bin/env python
# coding: utf-8
def filter_DF(df, filter):
    for i in filter:
        if i[1] == "==":
            df = df[df[i[0]]==i[2]]
        if i[1] == "!=":
            df = df[df[i[0]]!=i[2]]
        if i[1] == "<=":
            df = df[df[i[0]]<=i[2]]
        if i[1] == ">=":
            df = df[df[i[0]]>=i[2]]
        if i[1] == "<":
            df = df[df[i[0]]<i[2]]
        if i[1] == ">":
            df = df[df[i[0]]>i[2]]
    return df
# # 1. Imports/Options/Support functions

# ## 1.1 External imports

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import os
import time
import math
from sqlalchemy import create_engine
import gc
#import memory_profiler

# ## 1.2 Internal imports

# In[2]:


import sys
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)
# import the 'config' funtion from the config.py file

from configAWSRDS import config

# from support functions:

from Support_Functions import *


# In[3]:
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument("--client_name", type=str, required=True)
# Parse the argument
args = parser.parse_args()


# get config params
section = args.client_name
print("client name:", section)
filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
params = config(section=section,filename=filename)


# ## 1.3 Options

# In[4]:


# no max columns for pandas dataframe for a better visualization on jupyter notebooks
pd.set_option('display.max_columns', None)

# remove warnings
import warnings
warnings.filterwarnings('ignore')


# ## 1.4 Support functions

date = str(datetime.now().strftime("%Y%m%d"))

# In[5]:

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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# ## 2. Load data frames

# ### 2.1 Client related downloading of tables

# In[11]:


table_name = "transformed"
engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)

# In[12]:


productsxdayssceleton = import_data_AWSRDS(schema=table_name,table="shopify_productsxdays_sceleton",engine=engine)


# In[13]:


shopify_sales = import_data_AWSRDS(schema=table_name,table="shopify_sales",engine=engine)
shopify_sales = rename_columns(shopify_sales, productsxdayssceleton, prefix="shopify_")


# In[14]:


klaviyo = import_data_AWSRDS(schema=table_name,table="klaviyo",engine=engine)
klaviyo = rename_columns(klaviyo, productsxdayssceleton, prefix="klaviyo_")


# In[15]:


facebook_ads = import_data_AWSRDS(schema=table_name,table="facebook_ads",engine=engine)
facebook_ads = rename_columns(facebook_ads, productsxdayssceleton, prefix="facebook_ads_")


# In[16]:


google_analytics = import_data_AWSRDS(schema=table_name,table='"google_analytics"',engine=engine)
google_analytics = rename_columns(google_analytics, productsxdayssceleton, prefix="google_analytics_")

google_ads = import_data_AWSRDS(schema=table_name,table="google_ads",engine=engine)
google_ads = rename_columns(google_ads, productsxdayssceleton, prefix="google_ads_")


# In[17]:


marketingAndSales_plan = import_data_AWSRDS(schema=table_name,table="marketingandsales_plan",engine=engine)
marketingAndSales_plan = rename_columns(marketingAndSales_plan, productsxdayssceleton, prefix="marketingAndSales_plan_")


# In[18]:


engine.dispose()


# ### 2.2 Client non-related downloading of tables
# get config params
section = "external"
params = config(section=section,filename=filename)


# create sqlalchemy connection    
engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)
params


# In[20]:


table_name = "transformed"


# In[21]:


external_covid_data = import_data_AWSRDS(schema=table_name,table="external_covid_data",engine=engine)
external_covid_data = rename_columns(external_covid_data, productsxdayssceleton, prefix="external_covid_data_")

external_holidays_and_special_events_by_date = import_data_AWSRDS(schema=table_name,table="external_holidays_and_special_events_by_date",engine=engine)
external_holidays_and_special_events_by_date = rename_columns(external_holidays_and_special_events_by_date, productsxdayssceleton, prefix="external_holidays_and_special_events_by_date_")

external_weather = import_data_AWSRDS(schema=table_name,table="external_weather",engine=engine)
external_weather = rename_columns(external_weather, productsxdayssceleton, prefix="external_weather_")


# In[22]:


engine.dispose()


# #### Create list of all dataframes to be merged on sceleton

# In[23]:


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


# In[24]:


for df in all_dfs:
    print(df.shape)


# # 3. Rename features to categorize into Time Series Data Set variables

# In[25]:


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
    
"""
df = distinguish_features(df, productsxdayssceleton,
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])
"""

def get_dtypes(df, sceleton):
    dtype_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"feature", 0:"dtype"})
    for c in df.columns:
        if c in list(sceleton.columns):
            dtype_df = dtype_df[dtype_df.feature != c]
    dtype_df.reset_index(drop=True, inplace=True)
    return dtype_df


# In[26]:


# Remove all columns that exist in productxdayssceleton from to be merged dataframes except daydate and variant_id
for df in all_dfs_to_be_merged:
    drop_list = []
    print(df.columns[5])
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

# In[29]:


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

# In[30]:


klaviyo = distinguish_features(klaviyo, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = ["klaviyo_planned_recipients"],
                         time_varying_unknown_categoricals = [])


# ### facebook

# In[31]:


facebook_ads = distinguish_features(facebook_ads, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### google_analytics

# In[32]:


google_analytics = distinguish_features(google_analytics, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### google_ads

# In[33]:


google_ads = distinguish_features(google_ads, productsxdayssceleton, 
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# ### marketingAndSales_plan

# In[34]:


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

# In[35]:


external_covid_data = distinguish_features(external_covid_data, productsxdayssceleton,
                         static_categoricals = [],
                         static_reals = [],
                         time_varying_known_categoricals = [],
                         time_varying_known_reals = [],
                         time_varying_unknown_categoricals = [])


# In[36]:


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


# In[37]:


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

# In[38]:


# transform based on variable type
for df in all_dfs:
    for c in df.columns:
        if (c.startswith("SC_") or c.startswith("TVKC_") or c.startswith("TVUC_")):
            df[c].astype("category")
        if (c.startswith("SR_") or c.startswith("TVKR_") or c.startswith("TVUR_")):
            df[c].astype("float")


# In[39]:


for df in all_dfs:
    print(df.columns[5])
    df["TVKC_daydate"] = pd.to_datetime(df["TVKC_daydate"])


# # 6. Analysis of cutoff timings

# In[ ]:


# Checklist:
    # shopify_sales
    # google_analytics
    # google_ads
    # klaviyo
    # facebook_ads
    # marketingAndSales_plan
    # external_covid_data
    # external_weather
    # external_holidays_and_special_events_by_date


# In[57]:


df_results, df_results_min_analysis, df_results_max_analysis = get_minmax_above0_on_feature_level(external_holidays_and_special_events_by_date, "TVKC_daydate")


# In[58]:


df_results_max_analysis


# # 7. Merge

# #### Merge all scelecton based dataframes

# In[ ]:

#print(f'Memory usage report before merging with daydate and variant_id: {memory_profiler.memory_usage()}')

df_consolidated = merge_with_productsxdaysscelecton(productsxdayssceleton, all_dfs_to_be_merged_with_daydate_and_variantid, ["TVKC_daydate","SC_variant_id"])
df_consolidated.drop_duplicates()


# In[ ]:


print("Merged all dataframes with daydate and variant_id. Shape is now: ", df_consolidated.shape)
#print(f'Memory usage report after merging with daydate and variant_id: {memory_profiler.memory_usage()}')

# Free up memory
for df in all_dfs_to_be_merged_with_daydate_and_variantid:
    del df
del all_dfs_to_be_merged_with_daydate_and_variantid

# Call the garbage collector
gc.collect()

#print(f'Memory usage report after deleting all_dfs_to_be_merged_with_daydate_and_variantid dataframes: {memory_profiler.memory_usage()}')

# #### Merge all only daydate based dataframes

# In[ ]:

#print(f'Memory usage report before merging with daydate only: {memory_profiler.memory_usage()}')

df_consolidated = merge_with_productsxdaysscelecton(df_consolidated, all_dfs_to_be_merged_only_daydate, ["TVKC_daydate"])
df_consolidated.drop_duplicates()


# In[ ]:


print("Merged all dataframes with daydate only as well. Shape is now: ",df_consolidated.shape)
#print(f'Memory usage report after merging with daydate and variant_id: {memory_profiler.memory_usage()}')


# # 8. Temporal feature engineering

# ### Split in day, month, year and weekday

# In[ ]:


df_consolidated["year"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%Y')))
df_consolidated["month"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%m')))
df_consolidated["day"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%d')))
df_consolidated["weekday"] = df_consolidated['TVKC_daydate'].apply(lambda row: pd.to_datetime(row).isoweekday())


# ### add time_idx

# In[ ]:


# sort
df_consolidated.sort_values(["SC_variant_id","TVKC_daydate"], inplace=True)

# create counter to substract from index
products_df = pd.DataFrame(df_consolidated.SC_variant_id.unique(), columns=["SC_variant_id"])
products_df.insert(0, 'counter', value=np.arange(len(products_df)))
products_df

# generate idx and substract number of products * counter
df_consolidated = df_consolidated.merge(products_df, how="left", on="SC_variant_id")
df_consolidated.sort_values(["SC_variant_id","TVKC_daydate"], inplace=True)
df_consolidated.insert(0, 'time_idx', value=np.arange(len(df_consolidated)))

df_consolidated["time_idx"] = df_consolidated["time_idx"] - (len(df_consolidated.TVKC_daydate.unique()) * df_consolidated["counter"])

df_consolidated


# In[ ]:


df_consolidated.drop(columns="counter", inplace=True)


# ### add variable categories to new features

# In[ ]:


df_consolidated.rename(columns={"year":"TVKC_year",
                                "month":"TVKC_month",
                                "day":"TVKC_day",
                                "weekday":"TVKC_weekday",
                                "time_idx":"TVKR_time_idx"}, inplace=True)


# ### change dtype regarding variable categories of new features

# In[ ]:


for c in ["TVKC_year","TVKC_month","TVKC_day","TVKC_weekday"]:
    df_consolidated[c] = df_consolidated[c].astype("category")
    
df_consolidated["TVKR_time_idx"] = df_consolidated["TVKR_time_idx"].astype("int")


# # 9. Final checks and corrections

# ### check for nans

# In[ ]:


print("NaN values in df:", df_consolidated.isna().sum().sum())


# ### "." to "_" in feature names

# In[ ]:


for c in df_consolidated.columns:
    if "." in c:
        print("Renaming:", c, " to:", c.replace(".","_"))
        df_consolidated.rename(columns={c:c.replace(".","_")},inplace=True)


# replace inf values

df_consolidated.replace([np.inf, -np.inf], 0, inplace=True)

# # 10. Categorization of products for different TFT-trainings

# # In[ ]:


# # set parameters for filtering
# number_of_historic_days = 380

# # for TFT with normal sales quantty
# number_of_sales_top = number_of_historic_days*2 #100

# # for TFT with 7daysaverage
# number_of_sales_buttom = number_of_historic_days/2 #50

# # only for category analysis
# year_to_filter_by = 2022


# # In[ ]:


# # filter by category and year to split into categories
# categorization = df_consolidated.groupby(["SC_product_category","TVKC_year"], as_index=False).agg({"TVUR_shopify_lineitems_quantity":"sum",
#                                                                                         "SC_variant_id":"nunique",
#                                                                                         "SC_product_id":"nunique"})

# # calculate category ratio
# categorization["ratio"] = categorization["TVUR_shopify_lineitems_quantity"]/categorization["SC_variant_id"]

# categorization = categorization[categorization["TVKC_year"] == year_to_filter_by]
# categorization = categorization.sort_values(by="ratio", ascending=False)

# categorization


# # In[ ]:


# YoY = datetime.now() - timedelta(days=number_of_historic_days)

# categorization_variant_id = df_consolidated[pd.to_datetime(df_consolidated["TVKC_daydate"]) > YoY]

# categorization_variant_id = categorization_variant_id.groupby(["SC_product_category","SC_variant_id"], as_index=False)["TVUR_shopify_lineitems_quantity"].sum()

# categorization_variant_id = categorization_variant_id.sort_values(by="TVUR_shopify_lineitems_quantity", ascending=False)

# categorization_variant_id_top = categorization_variant_id[categorization_variant_id["TVUR_shopify_lineitems_quantity"] >= number_of_sales_top]
# categorization_variant_id_buttom = categorization_variant_id[(categorization_variant_id["TVUR_shopify_lineitems_quantity"] < number_of_sales_top) &
#                                                             (categorization_variant_id["TVUR_shopify_lineitems_quantity"] >= number_of_sales_buttom)]
# categorization_variant_id_kicked = categorization_variant_id[categorization_variant_id["TVUR_shopify_lineitems_quantity"] < number_of_sales_buttom]

# print("Number of top products above", number_of_sales_top, "in sales:", len(categorization_variant_id_top))
# print("Number of 7daysaverage products between", number_of_sales_top,"and",number_of_sales_buttom, "in sales:", len(categorization_variant_id_buttom))
# print("Number of to be kicked products below", number_of_sales_buttom, "in sales:", len(categorization_variant_id_kicked))


# # In[ ]:


# data_top = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_top.SC_variant_id))]
# data_buttom = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_buttom.SC_variant_id))]
# data_kicked = df_consolidated[df_consolidated["SC_variant_id"].isin(list(categorization_variant_id_kicked.SC_variant_id))]
from new_product_split_ import categorize_products
target =  "TVUR_shopify_lineitems_quantity"
try:
    print("Categorizing products")
    # Call to categorize_products to obtain the datasets
    results = categorize_products(param_store_path = "/product_analysis/config", days= 120, data=df_consolidated, target = target)
    print("Categorizing products Done")
    data_top = results['quantile_loss_data']
    data_buttom = results['tweedie_loss_data']
    data_kicked = results['naive_forecast_data']
except Exception as e:
    import traceback
    print(f"Error occurred: {e}")
    print(traceback.format_exc())


# # 11. Export

# ## Split top, buttom (long-tail) and kicked products

# In[ ]:


data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated.csv")
data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated.csv")
data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated.csv")
pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated.csv")


# In[ ]:


data_top = cutoff_beginning_of_timeseries(data_top, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")


# In[ ]:


data_buttom = cutoff_beginning_of_timeseries(data_buttom, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")


# In[ ]:


data_kicked = cutoff_beginning_of_timeseries(data_kicked, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")


# In[ ]:


data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated_cutoff.csv")
data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated_cutoff.csv")
data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated_cutoff.csv")
pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated_cutoff.csv")


# ## Whole dataset

# In[ ]:


df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated.csv")


# In[ ]:


df_consolidated = cutoff_beginning_of_timeseries(df_consolidated, 
                               target="TVUR_shopify_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")


# In[ ]:


df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated_cutoff.csv")

