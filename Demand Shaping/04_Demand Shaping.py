#!/usr/bin/env python
# coding: utf-8

# # 1. Imports/Options

# ## 1.1 External imports


# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import time
import os


# # 1.2 Internal imports

# In[2]:

# In[3]:
############# DATE VARIABLES ##########################
today = date.today()
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

from Support_Functions import *


# In[4]:

import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument("--client_name", type=str, required=True)
# Parse the argument
args = parser.parse_args()

# get config params
# get config params
section = args.client_name
print("client name:", section)
filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
params = config(section=section,filename=filename)



# create sqlalchemy connection
from sqlalchemy import create_engine    

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)


# ## 1.3 Options

# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)


# ## 1.4 Support functions

# In[5]:


def assess_status_over_levels_FB(campaign_status, adset_status):
    if (campaign_status == "ACTIVE") & (adset_status == "ACTIVE"):
        return True
    else:
        return False
    
def assess_status_over_levels_GA(campaign_status):
    if campaign_status == "ENABLED":
        return True   
    else:
        return False
    
def only_return_last_date(df, date_column, level):
    """
    for a chosen grouping level only give back the rows with the latest dates, e.g. useful for finding current campaign status
    """
    df.sort_values(by=[level, date_column], ascending=False, inplace=True)
    df_new = df.groupby([level], as_index=False)[date_column].first()
    df_new = df_new.merge(df, how="left", on=[level, date_column])
    return df_new

def get_last_sales_date(df):
    """
    from a dataframe retrieve the last day where sales (lineitems_quantity) happend
    """
    #last_sales_date = df.groupby("daydate", as_index=False)["lineitems_quantity"].sum().sort_values(by="daydate", ascending=False)
    #last_sales_date = last_sales_date[last_sales_date["lineitems_quantity"] != 0].daydate.iloc[0]
    """if last_sales_date.strftime("%Y-%m-%d") == pd.to_datetime(datetime.today()).strftime("%Y-%m-%d"):
        last_sales_date = last_sales_date - relativedelta(days=1)"""
    return pd.to_datetime(str(yesterday), utc=True) # pd.to_datetime(yesterday, utc=True) #last_sales_date - relativedelta(days=1)

def ffill_facebook(facebook_ads_spend_total, facebook_ads_adsets_daily_budget):
    """
    to use this in ffill of FB spends, if daily ad set budget is higher then spends, take the average. otherwise take the spends from yesterday
    """
    if facebook_ads_adsets_daily_budget > facebook_ads_spend_total:
        return (facebook_ads_adsets_daily_budget + facebook_ads_spend_total)/2
    else:
        return facebook_ads_spend_total
    
def roas_calculation(revenue, spend):
    """
    prevent running in error when performing division if spend is 0 
    """
    if spend == 0:
        return 0
    else:
        return (revenue / spend) * 100


# # 2. Load data

# ## Facebook Ad Spends

# In[6]:


facebook_ad_spends = import_data_AWSRDS(schema="demand_shaping",table="facebook_ad_spends",engine=engine)


# ## Google Ad Spends

# In[7]:


google_ad_spends = import_data_AWSRDS(schema="demand_shaping",table="google_ad_spends",engine=engine)


# ## Transformed data

# In[8]:


shopify_sales_transformed = import_data_AWSRDS(schema="transformed",table="shopify_sales",engine=engine)


# ## Plan data

# In[9]:


plan_data_transformed = import_data_AWSRDS(schema="transformed",table="marketingandsales_plan",engine=engine)


# ## Product sceleton

# In[10]:


sceleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)


# ## Forecasts

# In[11]:

#forecasts = import_data_AWSRDS(schema="forecasts",table="forecasts",engine=engine)
#forecasts_category = import_data_AWSRDS(schema="forecasts",table="forecasts_category",engine=engine)
forecasts = import_data_AWSRDS(schema="forecasts",table="forecasts_28days",engine=engine)
forecasts_category = import_data_AWSRDS(schema="forecasts",table="forecasts_28days_category",engine=engine)


# # 3. Create demand shaping dataframe

# ### Features
# 
# 1. Sales aka lineitems_quantity
#     1. lineitems_quantity -> take from transformed data and aggregate on category __DONE__
#     2. lineitems_quantity_forecasted -> take from inference and aggregate on category __FORECAST OUTSTANDING__
#     3. lineitems_quantity_target -> take from plan data and aggregate on category __DONE__
# 2. Price
#     1. lineitems_price -> take from transformed data and aggregate on category, future values already ffill with RRP __DONE__
# 3. Revenue    
#     1. revenue -> lineitems_price * lineitems_quantity __DONE__
#     2. revenue_forecasted -> lineitems_price * lineitems_quantity_forecasted __FORECAST OUTSTANDING__
#     3. revenue_target -> lineitems_price * lineitems_quantity_target __DONE__
# 4. Marketing Spends
#     1. Facebook Ads
#         1. FB_spend_historic -> take from FB demand shaping data and aggregate on category __DONE__
#         2. FB_spend_ffill -> take from yesterdays spend if adsets_daily_budget is NaN __DONE__
#     2. Google Ads
#         1. GA_spend_historic -> take from GA demand shaping data and aggregate on category __DONE__
#         2. GA_spend_ffill -> take from yesterdays spend __DONE__
#     3. Total spends
#         1. spends_total
#         2. spends_total_ffill
# 5. ROAS
#     1. ROAS_historic -> revenue / (spends_total)
#     2. ROAS_future -> revenue_forecasted / (spends_total_ffill)
#     4. ROAS_target -> revenue_target / (spends_total + spends_total_ffill)
# 6. Alerts
#     1. ROAS alert -> calculate if ROAS below 200%

# In[12]:


# set aggregator above individually, so you can easily switch to product
grouping_level = "product_category"


# ## 3.1 Create ROAS demand shaping basis frame

# In[13]:


# take shopify sales as basis

ROAS_demand_shaping = shopify_sales_transformed[['product_category_number',
                                                 'product_category',
                                                 'product_id',
                                                 'variant_sku',
                                                 'variant_id',
                                                 'daydate',
                                                 'lineitems_quantity',
                                                 'lineitems_price',
                                                 'rolling7days_lineitems_quantity',
                                                 'rolling30days_lineitems_quantity',
                                                 'rolling7days_lineitems_quantity_lastYear',
                                                 'rolling30days_lineitems_quantity_lastYear']].copy()

ROAS_demand_shaping


# ## 3.2 Engineer features on product level

# In[14]:


# engineer revenue
ROAS_demand_shaping["revenue"] = ROAS_demand_shaping["lineitems_quantity"] * ROAS_demand_shaping["lineitems_price"]


# ## 3.3 Engineer features for ad spend data

# ### 3.3.1 Facebook Ads

# In[15]:


# facebook ad spends on grouping level
facebook_demand_shaping = facebook_ad_spends.copy()

# divide adsets_daily_budget by 100 as it is in cents
facebook_demand_shaping["adsets_daily_budget"] = facebook_demand_shaping["adsets_daily_budget"] / 100

# groupby to get on adset level per day
facebook_demand_shaping = facebook_demand_shaping.groupby(["campaigns_id", "adsets_id", "date_start"], as_index=False)[["spend",
                                                                                                                        "adsets_daily_budget",
                                                                                                                        "mapping"]].agg({"spend":"sum",
                                                                                                                                         "adsets_daily_budget":"mean",
                                                                                                                                         "mapping":"first"})

# rename mapping to product_category
facebook_demand_shaping.rename(columns={"mapping":"product_category"}, inplace=True)


# In[16]:


# facebook ad status overview

# get only last dates
facebook_ads_status = facebook_ad_spends.copy()
facebook_ads_status = only_return_last_date(facebook_ads_status, "date_start", "adsets_id")

# create status column
facebook_ads_status["status"] = facebook_ads_status.apply(lambda x: assess_status_over_levels_FB(x["campaigns_effective_status"], x["adsets_effective_status"]), axis=1)
facebook_ads_status["status"] = facebook_ads_status["status"].astype("boolean")
facebook_ads_status = facebook_ads_status.groupby(["campaigns_id", "adsets_id", "date_start"], as_index=False)["status"].mean()

# rename
facebook_ads_status.rename(columns={"status":"current_status"}, inplace=True)

# drop
facebook_ads_status.drop(columns="date_start", inplace=True)


# In[17]:


# merge on facebook_demand_shaping
facebook_demand_shaping = facebook_demand_shaping.merge(facebook_ads_status, how="left", on=["campaigns_id","adsets_id"])

# rename dates column
facebook_demand_shaping.rename(columns={"date_start":"daydate"}, inplace=True)

# create a spend and budget column where current_status is active to use this spend later for ffill only (only consider currently active campaigns
facebook_demand_shaping["spend_ffill"] = facebook_demand_shaping["spend"] * facebook_demand_shaping["current_status"]
facebook_demand_shaping["adsets_daily_budget_ffill"] = facebook_demand_shaping["adsets_daily_budget"] * facebook_demand_shaping["current_status"]
facebook_demand_shaping


# ### 3.3.2 Google Ads

# In[18]:


# google ad spends on grouping level
google_ads_demand_shaping = google_ad_spends.copy()

# divide adsets_daily_budget by 1.000.000 as it is in micros
google_ads_demand_shaping["campaign_budget_amount_micros"] = google_ads_demand_shaping["campaign_budget_amount_micros"] / 1000000
google_ads_demand_shaping["metrics_cost_micros"] = google_ads_demand_shaping["metrics_cost_micros"] / 1000000

# groupby to get on adset level per day
google_ads_demand_shaping = google_ads_demand_shaping.groupby(["campaign_id", "segments_date"], as_index=False)[["metrics_cost_micros",
                                                                                                                        "campaign_budget_amount_micros",
                                                                                                                        "mapping"]].agg({"metrics_cost_micros":"sum",
                                                                                                                                         "campaign_budget_amount_micros":"mean",
                                                                                                                                         "mapping":"first"})
# rename mapping to product_category
google_ads_demand_shaping.rename(columns={"mapping":"product_category"}, inplace=True)


# In[19]:


# google ad status overview

# get only last dates
google_ads_status = google_ad_spends.copy()
google_ads_status = only_return_last_date(google_ads_status, "segments_date", "campaign_id")

# create status column
google_ads_status["status"] = google_ads_status.apply(lambda x: assess_status_over_levels_GA(x["campaign_status"]), axis=1)
google_ads_status["status"] = google_ads_status["status"].astype("boolean")
google_ads_status = google_ads_status.groupby(["campaign_id", "segments_date"], as_index=False)["status"].mean()

# rename
google_ads_status.rename(columns={"status":"current_status"}, inplace=True)

# drop
google_ads_status.drop(columns="segments_date", inplace=True)


# In[20]:


# merge on google_ads_demand_shaping
google_ads_demand_shaping = google_ads_demand_shaping.merge(google_ads_status, how="left", on=["campaign_id"])

# rename dates column
google_ads_demand_shaping.rename(columns={"segments_date":"daydate"}, inplace=True)

# rename budget and spends columns
google_ads_demand_shaping.rename(columns={"metrics_cost_micros":"spend"}, inplace=True)
google_ads_demand_shaping.rename(columns={"campaign_budget_amount_micros":"campaign_daily_budget"}, inplace=True)

# create a spend / budget column where current_status is active to use this spend later for ffill only (only consider currently active campaigns
google_ads_demand_shaping["spend_ffill"] = google_ads_demand_shaping["spend"] * google_ads_demand_shaping["current_status"]
google_ads_demand_shaping["campaign_daily_budget_ffill"] = google_ads_demand_shaping["campaign_daily_budget"] * google_ads_demand_shaping["current_status"]
google_ads_demand_shaping


# ### 3.3.3 Engineer features for plan data

# In[21]:


# get daily sales targets per cateogry
plan_data_transformed = plan_data_transformed.groupby(["product_category_number","daydate"], as_index=False)["daily_Product_Sales_Target"].mean()
plan_data_transformed.rename(columns={"daily_Product_Sales_Target":"lineitems_quantity_target"}, inplace=True)


# ## 3.4 Consolidate with other frames

# In[22]:


# convert all temporal features to same type
for df in [ROAS_demand_shaping, plan_data_transformed, google_ads_demand_shaping, facebook_demand_shaping]:
     df["daydate"] = pd.to_datetime(df["daydate"], utc=True)


# ### 3.4.1 Merge product level frames

# In[23]:


# merge plan data
ROAS_demand_shaping = ROAS_demand_shaping.merge(plan_data_transformed, how="left", on=["product_category_number", "daydate"])


# ### 3.4.2 Merge category level frames

# In[24]:


# group on category level for merge
ROAS_demand_shaping_catLevel = ROAS_demand_shaping.groupby(["product_category", "daydate"], as_index=False).agg({'lineitems_quantity':"sum",
                                                                                                                 'lineitems_price':"mean",
                                                                                                                 'rolling7days_lineitems_quantity':"sum",
                                                                                                                 'rolling30days_lineitems_quantity':"sum",
                                                                                                                 'rolling7days_lineitems_quantity_lastYear':"sum",
                                                                                                                 'rolling30days_lineitems_quantity_lastYear':"sum",
                                                                                                                 'revenue':"sum",
                                                                                                                 'lineitems_quantity_target':"mean"}).copy()


# In[25]:


# merge spend data on category level

# merge google_ads_demand_shaping

# group data accordingly
google_ads_demand_shaping_catLevel = google_ads_demand_shaping.groupby(["product_category","daydate"], as_index=False).agg({"spend":"sum", 
                                                                                                                             "spend_ffill":"sum", 
                                                                                                                             "campaign_daily_budget":"sum", 
                                                                                                                             "campaign_daily_budget_ffill":"sum", 
                                                                                                                             "current_status":"mean"})
google_ads_demand_shaping_catLevel.rename(columns={"spend":"google_ads_spend",
                                                   "spend_ffill":"google_ads_spend_ffill",
                                                   "campaign_daily_budget":"google_ads_campaign_daily_budget",
                                                   "campaign_daily_budget_ffill":"google_ads_campaign_daily_budget_ffill",
                                                   "current_status":"google_ads_current_status"}, inplace=True)

# perform merge
ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(google_ads_demand_shaping_catLevel, how="left", on=["product_category", "daydate"])

# merge facebook_demand_shaping

# group data accordingly
facebook_demand_shaping_catLevel = facebook_demand_shaping.groupby(["product_category","daydate"], as_index=False).agg({"spend":"sum",
                                                                                                                        "spend_ffill":"sum", 
                                                                                                                        "adsets_daily_budget":"sum", 
                                                                                                                        "adsets_daily_budget_ffill":"sum",
                                                                                                                        "current_status":"mean"})
facebook_demand_shaping_catLevel.rename(columns={"spend":"facebook_ads_spend", 
                                                 "spend_ffill":"facebook_ads_spend_ffill", 
                                                 "adsets_daily_budget":"facebook_ads_adsets_daily_budget",
                                                 "adsets_daily_budget_ffill":"facebook_ads_adsets_daily_budget_ffill",
                                                 "current_status":"facebook_ads_current_status"}, inplace=True)

# perform merge
ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(facebook_demand_shaping_catLevel, how="left", on=["product_category", "daydate"])

# fillna ROAS_demand_shaping_catLevel
ROAS_demand_shaping_catLevel.fillna(0, inplace=True)
ROAS_demand_shaping_catLevel


# ### 3.4.3 Break down "Rest" campaigns with weighted sales attribution

# In[26]:


# create sales weights per day

# 1. pivot sales per cateogry in columns using 7days rolling average
sales_by_category = ROAS_demand_shaping.groupby(["product_category", "daydate"], as_index=False)["rolling7days_lineitems_quantity"].sum()

sales_by_category = pd.pivot_table(sales_by_category, index=["daydate"], columns=["product_category"], aggfunc={"rolling7days_lineitems_quantity":"sum"}).reset_index()

sales_by_category.columns = sales_by_category.columns.get_level_values(1)
sales_by_category.rename(columns={"":"daydate"}, inplace=True)

# 2. create total sales column
sales_by_category["total_across_categories"] = sales_by_category[list(ROAS_demand_shaping.product_category.unique())].sum(axis=1)

# 3. get daily sales weights per category
#3.1 divide sales by total
for c in list(ROAS_demand_shaping.product_category.unique()):
    sales_by_category[c] = sales_by_category[c] / sales_by_category["total_across_categories"]
    sales_by_category.fillna(0, inplace=True)

# 3.2 incase everything is 0, put equal distribution
sales_by_category["all0"] = sales_by_category[list(ROAS_demand_shaping.product_category.unique())].sum(axis=1)
sales_by_category["all0"] = sales_by_category["all0"].apply(lambda x: False if x > 0 else True)

number_of_cats = ROAS_demand_shaping.product_category.nunique()

for c in list(ROAS_demand_shaping.product_category.unique()):
    sales_by_category[c] = np.where((sales_by_category["all0"] == True), (1/number_of_cats), sales_by_category[c])

# 4. get daily weighted spends per category

# 4.1 merge spends dataframes

# 4.1.1 for google ads

# create seperate dfs
google_ads_demand_shaping_catLevel_rest = google_ads_demand_shaping_catLevel[google_ads_demand_shaping_catLevel["product_category"] == "Rest"].copy()
google_ads_demand_shaping_catLevel_rest.rename(columns={"google_ads_spend":"google_ads_spend_rest", 
                                                        "google_ads_spend_ffill":"google_ads_spend_ffill_rest", 
                                                        "google_ads_campaign_daily_budget":"google_ads_campaign_daily_budget_rest", 
                                                        "google_ads_campaign_daily_budget_ffill":"google_ads_campaign_daily_budget_ffill_rest"}, inplace=True)
google_ads_demand_shaping_catLevel_rest.drop(columns=["product_category"], inplace=True)

# 4.1.2 for facebook
facebook_demand_shaping_catLevel_rest = facebook_demand_shaping_catLevel[facebook_demand_shaping_catLevel["product_category"] == "Rest"].copy()
facebook_demand_shaping_catLevel_rest.rename(columns={"facebook_ads_spend":"facebook_ads_spend_rest", 
                                                      "facebook_ads_spend_ffill":"facebook_ads_spend_ffill_rest", 
                                                      "facebook_ads_adsets_daily_budget":"facebook_ads_adsets_daily_budget_rest", 
                                                      "facebook_ads_adsets_daily_budget_ffill":"facebook_ads_adsets_daily_budget_ffill_rest"}, inplace=True)
facebook_demand_shaping_catLevel_rest.drop(columns=["product_category"], inplace=True)

# merge
sales_by_category = sales_by_category.merge(facebook_demand_shaping_catLevel_rest, how="left", on=["daydate"])



# merge
sales_by_category = sales_by_category.merge(google_ads_demand_shaping_catLevel_rest, how="left", on=["daydate"])

# 4.1.3 fill nan
sales_by_category.fillna(0, inplace=True)

# 4.2 multiply for each feature and unpivot

# 4.2.1 prepare features and empty df

# get features to transform
features = list(sales_by_category.columns)

for c in list(ROAS_demand_shaping.product_category.unique()):
    features.remove(c)
    
remove = ["daydate", "total_across_categories", "all0", "facebook_ads_current_status", "google_ads_current_status"]

for r in remove:
    features.remove(r)
    
# create empty df
weighted_spends_per_category = ROAS_demand_shaping[["daydate","product_category"]].drop_duplicates().copy()

# 4.2.2 execute transformation (multiplication and unpivoting per feature)
for f in features:
    # create sub df
    columns = ["daydate",f] + list(ROAS_demand_shaping.product_category.unique())
    sub = sales_by_category[columns].copy()

    # execute multiplication and drop feature after
    for c in list(ROAS_demand_shaping.product_category.unique()):
        sub[c] = sub[c] * sub[f]
    sub.drop(columns=f, inplace=True)

    # get list of all columns to melt
    c = list(sub.columns)
    c.remove("daydate")

    # melt aka unpivot dataframe
    sub = pd.melt(sub, id_vars=['daydate'], value_vars=c)
    
    # rename columns
    sub.rename(columns={"variable":"product_category", "value":f+"_weighted"}, inplace=True)
    
    # merge on weighted_spends_per_category
    weighted_spends_per_category = weighted_spends_per_category.merge(sub, how="left", on=["product_category", "daydate"])

# 5. merge on ROAS_demand_shaping_catLevel
ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(weighted_spends_per_category, how="left", on=["product_category", "daydate"])
ROAS_demand_shaping_catLevel


# ### 3.4.4 Create total spend columns

# In[27]:


# create total columns for spends

# change dtypes of ffill features to float
for f in ["facebook_ads_spend_ffill_rest_weighted", "google_ads_spend_ffill_rest_weighted", "facebook_ads_adsets_daily_budget_ffill_rest_weighted", "google_ads_campaign_daily_budget_ffill_rest_weighted"]:
    ROAS_demand_shaping_catLevel[f] = ROAS_demand_shaping_catLevel[f].astype("float")
    
# new features list
for i in range(len(features)):
    features[i] = features[i].replace("_rest","")

for f in features:
    ROAS_demand_shaping_catLevel[f+"_total"] = ROAS_demand_shaping_catLevel[f] + ROAS_demand_shaping_catLevel[f+"_rest_weighted"]


# ### 3.4.5 Forward fill spends for Facebook and Google Ads

# #### Facebook - using adset budgets as well

# In[28]:


# ffill facebook ad spends
to_merge = ROAS_demand_shaping_catLevel[ROAS_demand_shaping_catLevel["daydate"] == get_last_sales_date(ROAS_demand_shaping)][["product_category", "facebook_ads_spend_ffill_total", "facebook_ads_adsets_daily_budget_ffill_total"]].copy()
    
to_merge["facebook_spends_total_ffilled"] = to_merge.apply(lambda x: ffill_facebook(x["facebook_ads_spend_ffill_total"], x["facebook_ads_adsets_daily_budget_ffill_total"]), axis=1)
to_merge.drop(columns=["facebook_ads_spend_ffill_total", "facebook_ads_adsets_daily_budget_ffill_total"], inplace=True)

ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(to_merge, how="left", on="product_category")
ROAS_demand_shaping_catLevel


# #### Google - only using spends

# In[29]:


# ffill google ad spends, as google is a pull ads market, budget is not used

# sort ascending
ROAS_demand_shaping_catLevel.sort_values(by=["product_category", "daydate"], inplace=True)

# create to merge df
to_merge = ROAS_demand_shaping_catLevel[(ROAS_demand_shaping_catLevel["daydate"] <= get_last_sales_date(ROAS_demand_shaping))&
                                        (ROAS_demand_shaping_catLevel["daydate"] >= get_last_sales_date(ROAS_demand_shaping) -  relativedelta(days=3))][["daydate", 
                                                                                                                                                         "product_category", 
                                                                                                                                                         "google_ads_spend_ffill_total"]].copy()

# calculate average 4 day google ads spends

# 1. generate weights
to_merge_new = pd.DataFrame(columns=to_merge.columns)

for c in to_merge.product_category.unique():
    to_merge_new = pd.concat([to_merge_new, pd.concat([to_merge[to_merge["product_category"] == c].reset_index(drop=True), pd.DataFrame([1,2,3,4], columns=["weights"])], axis=1)])

# 2. multiply with weights and divide by sum of weights
to_merge_new["google_ads_spend_ffill_total"] = to_merge_new["google_ads_spend_ffill_total"] * to_merge_new["weights"]

# 3. groupby with weighted average
to_merge_new = to_merge_new.groupby(["product_category"], as_index=False).agg({"google_ads_spend_ffill_total":"sum","weights":"sum"})
to_merge_new["google_ads_spend_total_ffilled"] = to_merge_new["google_ads_spend_ffill_total"] / to_merge_new["weights"]
to_merge_new.drop(columns=["google_ads_spend_ffill_total", "weights"], inplace=True)

# merge on ROAS_demand_shaping_catLevel
ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(to_merge_new, how="left", on="product_category")
ROAS_demand_shaping_catLevel


# ### 3.4.5 Calculate total spends columns adding Google and Facebook up

# In[30]:


# total of all spends
ROAS_demand_shaping_catLevel["spends_total"] = ROAS_demand_shaping_catLevel["google_ads_spend_total"] + ROAS_demand_shaping_catLevel["facebook_ads_spend_total"]
ROAS_demand_shaping_catLevel["spends_total_ffill"] = ROAS_demand_shaping_catLevel["google_ads_spend_total_ffilled"] + ROAS_demand_shaping_catLevel["facebook_spends_total_ffilled"]


# ## 3.5 Integrate product forecasts

# In[31]:


forecasts_demand_shaping = forecasts.copy()

# calculate revenue
forecasts_demand_shaping["revenue_forecast"] = forecasts_demand_shaping["TVKR_variant_RRP"] * forecasts_demand_shaping["NOGL_forecast_q3"]

# select only necessary features
forecasts_demand_shaping = forecasts_demand_shaping[["TVKC_daydate", 
                                                     "SC_product_category",
                                                     "NOGL_forecast_q3",
                                                     "revenue_forecast"]]

# rename for easier merging
forecasts_demand_shaping.rename(columns={"TVKC_daydate":"daydate", 
                                         "SC_product_category":"product_category",
                                         "NOGL_forecast_q3":"lineitems_quantity_forecast"}, inplace=True)

# change dtype of daydate
forecasts_demand_shaping["daydate"] = pd.to_datetime(forecasts_demand_shaping["daydate"], utc=True)

# groupby category
forecasts_demand_shaping = forecasts_demand_shaping.groupby(["product_category", "daydate"], as_index = False)[["lineitems_quantity_forecast", "revenue_forecast"]].sum()


# In[32]:


# merge on ROAS_demand_shaping_catLevel
ROAS_demand_shaping_catLevel = ROAS_demand_shaping_catLevel.merge(forecasts_demand_shaping, how="left", on=["product_category", "daydate"])
ROAS_demand_shaping_catLevel


# ## 3.6 Engineer final features on consolidated frame

# In[33]:


# revenue_target
ROAS_demand_shaping_catLevel["revenue_target"] = ROAS_demand_shaping_catLevel["lineitems_quantity_target"] * ROAS_demand_shaping_catLevel["lineitems_price"]


# ## 3.7 Calculate ROAS

# ### 3.7.1 Select final features

# In[34]:


ROAS_demand_shaping_final = ROAS_demand_shaping_catLevel[['product_category',
                                                         'daydate',
                                                         'lineitems_quantity',
                                                         'lineitems_quantity_forecast', 
                                                         'lineitems_quantity_target',
                                                         'rolling7days_lineitems_quantity_lastYear',
                                                         'revenue',
                                                         'revenue_forecast',
                                                         'revenue_target',
                                                         'spends_total',
                                                         'spends_total_ffill']].copy()


# ### 3.7.2 Set parts of the dataframe zero depending on history or future

# In[35]:


# combine spends_total and spends_total_ffill to calculate ROAS_target
ROAS_demand_shaping_final_history = ROAS_demand_shaping_final[ROAS_demand_shaping_final["daydate"] <= get_last_sales_date(ROAS_demand_shaping)].copy()
ROAS_demand_shaping_final_future = ROAS_demand_shaping_final[ROAS_demand_shaping_final["daydate"] > get_last_sales_date(ROAS_demand_shaping)].copy()

history_and_future_dict = {'product_category':"both",
                           'daydate':"both",
                           'lineitems_quantity':"history",
                           'lineitems_quantity_forecast':"future", 
                           'lineitems_quantity_target':"both",
                           'rolling7days_lineitems_quantity_lastYear':"both",
                           'revenue':"history",
                           'revenue_forecast':"future",
                           'revenue_target':"both",
                           'spends_total':"history",
                           'spends_total_ffill':"future"}

# fill future or history around last sales event date with 0 depending on feature
for f in history_and_future_dict:
    if history_and_future_dict.get(f) == "future":
        ROAS_demand_shaping_final_history[f] = 0 # set below last sales date to 0
    if history_and_future_dict.get(f) == "history":
        ROAS_demand_shaping_final_future[f] = 0 # set above last sales date to 0

# concat ROAS_demand_shaping_final_history and ROAS_demand_shaping_final_future
ROAS_demand_shaping_final = pd.concat([ROAS_demand_shaping_final_history, ROAS_demand_shaping_final_future])
ROAS_demand_shaping_final.sort_values(by=["product_category", "daydate"], inplace=True)

ROAS_demand_shaping_final


# In[36]:


# combine spends
ROAS_demand_shaping_final["spends_total_combined"] = ROAS_demand_shaping_final["spends_total"] + ROAS_demand_shaping_final["spends_total_ffill"]


# ### 3.7.3 Calculate ROAS

# In[37]:


ROAS_demand_shaping_final["roas_historic"] = ROAS_demand_shaping_final.apply(lambda x: roas_calculation(x["revenue"], x["spends_total"]), axis=1)
ROAS_demand_shaping_final["roas_forecasted"] = ROAS_demand_shaping_final.apply(lambda x: roas_calculation(x["revenue_forecast"], x["spends_total_ffill"]), axis=1)
ROAS_demand_shaping_final["roas_target"] = ROAS_demand_shaping_final.apply(lambda x: roas_calculation(x["revenue_target"], x["spends_total_combined"]), axis=1)
ROAS_demand_shaping_final


# # TODO:
# - get active campaigns / adsets per category
# - include planned budgets -> Need breakdown on daily level?
# - Export: update df, do not overwrite

# # 4. Calculate alerts

# ## 4.1 Supply constraints

# In[38]:


# integrate daily capacity constraints on category level
germany_factor = 0.7
capacities = [["Tasse", (2500 * germany_factor)],
              ["T-Shirt", (1250 * germany_factor)],
              ["Handyhülle", (1500 * germany_factor)],
              ["Kissen", (750 * germany_factor)]]

# create DF of dict
capacities_df = pd.DataFrame(capacities, columns=["product_category", "capacity_constraint"])

# merge on ROAS_demand_shaping_final
ROAS_demand_shaping_final = ROAS_demand_shaping_final.merge(capacities_df, how="left", on="product_category")
ROAS_demand_shaping_final.fillna(0, inplace=True)

ROAS_demand_shaping_final


# ## 4.2 ROAS constraints

# In[39]:


# create border dataframe
ROAS_demand_shaping_final["ROAS_border"] = 200
ROAS_demand_shaping_final


# # 5. Export

# In[40]:


action_tracking = pd.DataFrame(["1", pd.to_datetime("2022-11-23"), 
                                "Facebook", 
                                "decrease budget for campaign XXX", 
                                "ROAS below 200% predicted by NOGL", 
                                "increase in profitability on product level", 
                                "n.a."],
                               ).transpose().rename(columns={0: "id", 1:"date", 2:"channel/campaign/ad", 3:"action", 4:"reason", 5:"expected outcome", 6:"actual outcome"})
action_tracking.to_sql("action_tracking", con = engine, schema="demand_shaping", if_exists='replace', index=False, chunksize=1000, method="multi")


# ## AWS RDS

# In[41]:


engine.dispose()


# In[42]:


engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True


# In[43]:


t = Timer("Upload")
ROAS_demand_shaping_final.to_sql("demandshaping_adspends", con = engine, schema="demand_shaping", if_exists='replace', index=False, chunksize=1000, method="multi")
t.end()


# In[44]:


engine.dispose()


# In[45]:


print(params)
