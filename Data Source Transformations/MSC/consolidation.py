import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from new_product_split import categorize_products
import time
from typing import Optional
from utils import *
import logging
import os
import datetime
import argparse
import time
from id_detection import *
from Support_Functions import *
import math


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

parser = argparse.ArgumentParser(description="This script creates product info table in database with unique nogl id.")

CLIENT_NAME = "dogs-n-tiger"
CHANNEL = "amazon"

parser.add_argument(
    "--client_name",
    type=str,
    default=CLIENT_NAME,
    help="Client name",
)

parser.add_argument(
    "--channel",
    type=str,
    default=CHANNEL,
    help="channel",
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
        
        # Convert datetime columns in both dataframes to timezone-naive
        for key in keys:
            if base[key].dtype == 'datetime64[ns, UTC]':
                df_merged[key] = df_merged[key].dt.tz_localize(None)
            if df[key].dtype == 'datetime64[ns, UTC]':
                df[key] = df[key].dt.tz_localize(None)
        
        df_merged = df_merged.merge(df, how="left", on=keys)
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



def consolidation(client_name, channel):

    if channel == "shopify":
        drop_channel = "amazon"
    else:
        drop_channel = "shopify"

    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name),
        "external" : get_db_connection(dbtype="prod", client_name="external")
    }

    products_dicts = {}
    with log_time("Importing data from AWS RDS"):

        table_name = "transformed"
        engine = engines["dev"]
        external_prod_engine = engines["external"]
        productsxdayssceleton = import_data_AWSRDS(schema="product",table="productsxdays_sceleton",engine=engine)
        # just extract the columns where they contain "channel" name and additionally daydate and nogl_id
        productsxdayssceleton = productsxdayssceleton[["nogl_id","daydate"]+[c for c in productsxdayssceleton.columns if channel in c]]
        # remove channel name from the columns if they contains
        productsxdayssceleton.columns = [c.replace(channel+"_","") for c in productsxdayssceleton.columns]
        productsxdayssceleton = productsxdayssceleton[productsxdayssceleton["variant_barcode"].notnull() | productsxdayssceleton["variant_sku"].notnull()]
        productsxdayssceleton.reset_index(drop=True,inplace=True)
        # rename variant_id to channel_variant_id and nogl_id to variant_id
        productsxdayssceleton.rename(columns={"nogl_id":"variant_id"}, inplace=True)
        if channel == "shopify":
            shopify_sales = import_data_AWSRDS(schema=table_name,table="msc_shopify_sales",engine=engine)
            shopify_sales.rename(columns={"nogl_id":"variant_id"}, inplace=True)
            shopify_sales = rename_columns(shopify_sales, productsxdayssceleton, prefix="shopify_")
        else:
            amazon_sales = import_data_AWSRDS(schema=table_name,table="msc_amazon_sales",engine=engine)
            amazon_sales.rename(columns={"nogl_id":"variant_id"}, inplace=True)
            amazon_sales = rename_columns(amazon_sales, productsxdayssceleton, prefix="amazon_")


        klaviyo = import_data_AWSRDS(schema=table_name,table="msc_klaviyo",engine=engine)
        # drop the columns that contains drop_channel 
        klaviyo = klaviyo[[c for c in klaviyo.columns if drop_channel not in c]]
        # remove channel prefx from the columns if they contains as substring
        klaviyo.columns = [c.replace(channel+"_","") for c in klaviyo.columns]
        klaviyo = rename_columns(klaviyo, productsxdayssceleton, prefix="klaviyo_")

        google_analytics = import_data_AWSRDS(schema=table_name,table="msc_google_analytics",engine=engine)
        # drop the columns that contains drop_channel 
        google_analytics = google_analytics[[c for c in google_analytics.columns if drop_channel not in c]]
        # remove channel prefx from the columns if they contains as substring
        google_analytics.columns = [c.replace(channel+"_","") for c in google_analytics.columns]
        google_analytics = rename_columns(google_analytics, productsxdayssceleton, prefix="google_analytics_")

        google_ads = import_data_AWSRDS(schema=table_name,table="msc_google_ads",engine=engine)
        # drop the columns that contains drop_channel 
        google_ads = google_ads[[c for c in google_ads.columns if drop_channel not in c]]
        # remove channel prefx from the columns if they contains as substring
        google_ads.columns = [c.replace(channel+"_","") for c in google_ads.columns]
        google_ads = rename_columns(google_ads, productsxdayssceleton, prefix="google_ads_")


        facebook_ads = import_data_AWSRDS(schema=table_name,table="msc_facebook_ads",engine=engine)
        # drop the columns that contains drop_channel 
        facebook_ads = facebook_ads[[c for c in facebook_ads.columns if drop_channel not in c]]
        # remove channel prefx from the columns if they contains as substring
        facebook_ads.columns = [c.replace(channel+"_","") for c in facebook_ads.columns]
        facebook_ads = rename_columns(facebook_ads, productsxdayssceleton, prefix="facebook_ads_")

        marketingAndSales_plan = import_data_AWSRDS(schema="product",table="plan_data",engine=engine)
        # drop the columns that contains drop_channel 
        marketingAndSales_plan = marketingAndSales_plan[[c for c in marketingAndSales_plan.columns if drop_channel not in c]]
        # remove channel prefx from the columns if they contains as substring
        marketingAndSales_plan.columns = [c.replace(channel+"_","") for c in marketingAndSales_plan.columns]
        marketingAndSales_plan = rename_columns(marketingAndSales_plan, productsxdayssceleton, prefix="marketingAndSales_plan_")

        external_covid_data = import_data_AWSRDS(schema="transformed",table="external_covid_data",engine=external_prod_engine)
        external_covid_data = rename_columns(external_covid_data, productsxdayssceleton, prefix="external_covid_data_")


        external_holidays_and_special_events_by_date = import_data_AWSRDS(schema="transformed",table="external_holidays_and_special_events_by_date",engine=external_prod_engine)
        external_holidays_and_special_events_by_date = rename_columns(external_holidays_and_special_events_by_date, productsxdayssceleton, prefix="external_holidays_and_special_events_by_date_")

        external_weather = import_data_AWSRDS(schema="transformed",table="external_weather",engine=external_prod_engine)
        external_weather = rename_columns(external_weather, productsxdayssceleton, prefix="external_weather_")



        all_dfs_to_be_merged = [#shopify_sales,
                        klaviyo,
                        facebook_ads,
                        google_analytics,
                        google_ads,
                        marketingAndSales_plan,
                        external_covid_data,
                        external_holidays_and_special_events_by_date,
                        external_weather]

        all_dfs_to_be_merged_with_daydate_and_variantid = [#shopify_sales,
                                klaviyo,
                                facebook_ads,
                                google_analytics,
                                google_ads,
                                marketingAndSales_plan]

        all_dfs_to_be_merged_only_daydate = [external_covid_data,
                                external_holidays_and_special_events_by_date,
                                external_weather]



        
        if channel == "shopify":
            all_dfs_to_be_merged.append(shopify_sales)
            all_dfs_to_be_merged_with_daydate_and_variantid.append(shopify_sales)
        else:
            all_dfs_to_be_merged.append(amazon_sales)
            all_dfs_to_be_merged_with_daydate_and_variantid.append(amazon_sales)
            
        all_dfs = all_dfs_to_be_merged.copy()
        all_dfs.append(productsxdayssceleton)


        for df in all_dfs:
            print(df.shape)


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

        #productsxdayssceleton.drop(columns=["variant_created_at", "product_published_at", "variant_max_updated_at"], inplace=True)
        if channel == "shopify":
            shopify_sales.drop(columns=["shopify_lineitems_sku","shopify_lineitems_variant_inventory_management"], inplace=True)

            productsxdayssceleton = distinguish_features(productsxdayssceleton, productsxdayssceleton,
                                                static_categoricals = ["product_category_number",
                                                                        "product_category",
                                                                        "product_id",
                                                                        "variant_sku",
                                                                        "variant_barcode",
                                                                        "variant_id", # nogl_id
                                                                        "channel_variant_id", # shopify_variant_id
                                                                        #"variant_inventory_item_id",
                                                                        #"variant_created_at", # DROPPED
                                                                        #"product_published_at", # DROPPED
                                                                        "product_published_scope"],
                                                static_reals = [],
                                                time_varying_known_categoricals = ["daydate", 
                                                                                    "year",
                                                                                    "month",
                                                                                    "day",
                                                                                    "weekday",
                                                                                    #"variant_taxable",
                                                                                    #"variant_requires_shipping",
                                                                                    "product_status",
                                                                                    #"variant_inventory_management_used", 
                                                                                    #"variant_max_updated_at" # DROPPED
                                                                                    ],
                                                time_varying_known_reals = ["variant_RRP"],
                                                time_varying_unknown_categoricals = [])
        else:

            productsxdayssceleton = distinguish_features(productsxdayssceleton, productsxdayssceleton,
                                                static_categoricals = ["product_category_number",
                                                                        "product_category",
                                                                        "product_id",
                                                                        "variant_sku",
                                                                        "variant_barcode",
                                                                        "variant_asin",
                                                                        "variant_id", # nogl_id
                                                                        #"variant_inventory_item_id",
                                                                        #"variant_created_at", # DROPPED
                                                                        #"product_published_at", # DROPPED
                                                                        "product_published_scope"],
                                                static_reals = [],
                                                time_varying_known_categoricals = ["daydate", 
                                                                                    "year",
                                                                                    "month",
                                                                                    "day",
                                                                                    "weekday",
                                                                                    #"variant_taxable",
                                                                                    #"variant_requires_shipping",
                                                                                    "product_status",
                                                                                    #"variant_inventory_management_used", 
                                                                                    #"variant_max_updated_at" # DROPPED
                                                                                    ],
                                                time_varying_known_reals = ["variant_RRP"],
                                                time_varying_unknown_categoricals = [])



        if channel == "shopify":    

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

        else:

            amazon_sales = distinguish_features(amazon_sales, productsxdayssceleton,
                                                static_categoricals = [#"shopify_lineitems_sku" # DROPPED
                                                                    ],
                                                static_reals = [],
                                                time_varying_known_categoricals = [],
                                                time_varying_known_reals = ["amazon_lineitems_price",
                                                                            "amazon_lineitems_variant_inventory_management",
                                                                            "amazon_orders_currency_EUR",
                                                                            "amazon_orders_presentment_currency_EUR",
                                                                            "amazon_orders_source_web",
                                                                            "amazon_rolling7days_lineitems_quantity_lastYear",
                                                                            "amazon_rolling30days_lineitems_quantity_lastYear",
                                                                            "amazon_rolling7days_lineitems_quantity_trend_lastYear",
                                                                            "amazon_rolling7days_lineitems_quantity_season_lastYear",
                                                                            "amazon_rolling7days_lineitems_quantity_resid_lastYear"],
                                                time_varying_unknown_categoricals = [
                                                                                    "amazon_product_condition",
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


        for df in all_dfs:
            for c in df.columns:
                print(c)
                if (c.startswith("SC_") or c.startswith("TVKC_") or c.startswith("TVUC_")):
                    df[c].astype("category")
                if (c.startswith("SR_") or c.startswith("TVKR_") or c.startswith("TVUR_")):
                    df[c].astype("float")

        for df in all_dfs:
            print(df.columns[5])
            df["TVKC_daydate"] = pd.to_datetime(df["TVKC_daydate"])


        df_results, df_results_min_analysis, df_results_max_analysis = get_minmax_above0_on_feature_level(external_holidays_and_special_events_by_date, "TVKC_daydate")
        df_consolidated = merge_with_productsxdaysscelecton(productsxdayssceleton, all_dfs_to_be_merged_with_daydate_and_variantid, ["TVKC_daydate","SC_variant_id"])
        df_consolidated.drop_duplicates()


        df_consolidated = merge_with_productsxdaysscelecton(df_consolidated, all_dfs_to_be_merged_only_daydate, ["TVKC_daydate"])
        df_consolidated.drop_duplicates()


        print("Merged all dataframes with daydate and variant_id. Shape is now: ", df_consolidated.shape)


        df_consolidated["year"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%Y')))
        df_consolidated["month"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%m')))
        df_consolidated["day"] = df_consolidated['TVKC_daydate'].apply(lambda row: int(row.strftime('%d')))
        df_consolidated["weekday"] = df_consolidated['TVKC_daydate'].apply(lambda row: pd.to_datetime(row).isoweekday())

        #show duplicates in SC_variant_id and daydate
        df_consolidated[df_consolidated.duplicated(subset=["SC_variant_id","TVKC_daydate"], keep=False)].sort_values(by=["SC_variant_id","TVKC_daydate"])

        # drop duplicates in SC_variant_id and daydate
        df_consolidated = df_consolidated.drop_duplicates(subset=["SC_variant_id","TVKC_daydate"], keep="first")

        df_consolidated.sort_values(["SC_variant_id","TVKC_daydate"], inplace=True)





        products_df = pd.DataFrame(df_consolidated.SC_variant_id.unique(), columns=["SC_variant_id"])
        products_df.insert(0, 'counter', value=np.arange(len(products_df)))
        products_df

        # generate idx and substract number of products * counter
        df_consolidated = df_consolidated.merge(products_df, how="left", on="SC_variant_id")
        df_consolidated.sort_values(["SC_variant_id","TVKC_daydate"], inplace=True)
        df_consolidated.insert(0, 'time_idx', value=np.arange(len(df_consolidated)))


        df_consolidated["time_idx"] = df_consolidated["time_idx"] - (len(df_consolidated.TVKC_daydate.unique()) * df_consolidated["counter"])

        # show duplicates in time_idx
        df_consolidated[df_consolidated.duplicated(subset=["time_idx"], keep=False)].sort_values(by=["time_idx"])



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


        print("NaN values in df:", df_consolidated.isna().sum().sum())

        for c in df_consolidated.columns:
            if "." in c:
                print("Renaming:", c, " to:", c.replace(".","_"))
                df_consolidated.rename(columns={c:c.replace(".","_")},inplace=True)


        print("NaN values in df:", df_consolidated.isna().sum().sum())


        # replace inf values
        df_consolidated.replace([np.inf, -np.inf], 0, inplace=True)
        # # set parameters for filtering
        # number_of_historic_days = 380
        # # for TFT with normal sales quantty
        # number_of_sales_top = number_of_historic_days*2 #100
        # # for TFT with 7daysaverage
        # number_of_sales_buttom = number_of_historic_days/2 #50
        # # only for category analysis
        # year_to_filter_by = 2022

        # TODO : conver type to float
        df_consolidated["TVKR_variant_RRP"] = df_consolidated["TVKR_variant_RRP"].astype(float)


        try:
            print("Categorizing products")
            # Call to categorize_products to obtain the datasets
            results = categorize_products(param_store_path = "/product_analysis/config", days= 365, data=df_consolidated)
            print("Categorizing products Done")
            data_top = results['quantile_loss_data']
            data_buttom = results['tweedie_loss_data']
            data_kicked = results['naive_forecast_data']
        except Exception as e:
            import traceback
            print(f"Error occurred: {e}")
            print(traceback.format_exc())

        # show duplicates in TVKR_time_idx
        print(df_consolidated[df_consolidated.duplicated(subset=["TVKR_time_idx"], keep=False)].sort_values("TVKR_time_idx"))
        
        # get current date
        # get current date
        date = datetime.now().strftime("%Y%m%d")

        data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated_"+channel+".csv")
        data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated_"+channel+".csv")
        data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated_"+channel+".csv")
        pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated_"+channel+".csv")


        data_top = cutoff_beginning_of_timeseries(data_top, 
                               target="TVUR_"+channel+"_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")

        data_buttom = cutoff_beginning_of_timeseries(data_buttom, 
                                    target="TVUR_"+channel+"_lineitems_quantity", 
                                    time_idx="TVKR_time_idx", 
                                    identifier="SC_variant_id")

        data_kicked = cutoff_beginning_of_timeseries(data_kicked, 
                                    target="TVUR_"+channel+"_lineitems_quantity", 
                                    time_idx="TVKR_time_idx", 
                                    identifier="SC_variant_id")
        

        data_top.to_csv("/opt/ml/processing/output/"+date+"_TopSeller_consolidated_cutoff_"+channel+".csv")
        data_buttom.to_csv("/opt/ml/processing/output/"+date+"_LongTail_consolidated_cutoff_"+channel+".csv")
        data_kicked.to_csv("/opt/ml/processing/output/"+date+"_Kicked_consolidated_cutoff_"+channel+".csv")
        pd.concat([data_top, data_buttom]).to_csv("/opt/ml/processing/output/"+date+"_TopAndLong_consolidated_cutoff_"+channel+".csv")

        df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated"+channel+".csv")

        df_consolidated = cutoff_beginning_of_timeseries(df_consolidated, 
                               target="TVUR_"+channel+"_lineitems_quantity", 
                               time_idx="TVKR_time_idx", 
                               identifier="SC_variant_id")
        
        df_consolidated.to_csv("/opt/ml/processing/output/"+date+"_Total_consolidated_cutoff"+channel+".csv")


if __name__ == "__main__":
    consolidation(args.client_name, args.channel)