# Imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from contextlib import contextmanager
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

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

parser = argparse.ArgumentParser(description="This script creates product info table in database with unique nogl id.")

CLIENT_NAME = "dogs-n-tiger"

parser.add_argument(
    "--client_name",
    type=str,
    default=CLIENT_NAME,
    help="Client name",
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

def load_campaignperformance(engine, incl_name=False):
    """
    Loads campaign performance
    Arguments:
        name, .csv file name to load
    """
    # load data
    df = import_data_AWSRDS(schema="googleads",table="custom_campaignperformance_scd",engine=engine)

    # select columns
    df = df[[
        'segments.date',
        'campaign.id',
        'campaign.name', # probably irrelevant
        'campaign.status',
        'campaign.start_date',
        'campaign.end_date',
        'campaign.advertising_channel_type',
        #'campaign.advertising_channel_sub_type', # KICK OUT -> No real differentiation
        #'campaign.bidding_strategy', # KICK OUT -> Only nan values
        #'campaign.campaign_budget', #  KICK OUT -> Not relevant info
        'campaign.bidding_strategy_type',
        'campaign_budget.amount_micros', # from here on METRICS
        'campaign_budget.reference_count', # description: number of campaigns actively using the budget
        'metrics.ctr',
        'metrics.clicks',
        'metrics.average_cpc',
        'metrics.average_cpe',
        'metrics.average_cpm',
        'metrics.average_cpv',
        'metrics.bounce_rate',
        'metrics.conversions',
        'metrics.cost_micros',
        'metrics.engagements',
        #'metrics.gmail_saves', #  KICK OUT -> Almost all ZEROS
        'metrics.impressions',
        'metrics.video_views',
        'metrics.average_cost',
        'metrics.interactions',
        'metrics.relative_ctr',
        #'metrics.gmail_forwards', #  KICK OUT -> Only ZEROS
        'metrics.engagement_rate',
        'metrics.video_view_rate',
        'metrics.interaction_rate',
        'metrics.conversions_value',
        'metrics.average_page_views',
        'metrics.search_click_share',
        'metrics.cost_per_conversion',
        'metrics.average_time_on_site',
        'metrics.percent_new_visitors',
        'metrics.value_per_conversion',
        #'metrics.gmail_secondary_clicks', # KICK OUT -> Almost all ZEROS
        'metrics.search_impression_share',
        'metrics.video_quartile_p25_rate',
        'metrics.video_quartile_p50_rate',
        'metrics.video_quartile_p75_rate',
        'metrics.content_impression_share',
        'metrics.video_quartile_p100_rate',
        'metrics.view_through_conversions',
        'metrics.top_impression_percentage',
        'metrics.search_top_impression_share',
        'metrics.absolute_top_impression_percentage',
        'metrics.conversions_from_interactions_rate',
        'metrics.search_absol___top_impression_share'
    ]]
    
    if incl_name == False:
        df.drop(columns="campaign.name", inplace=True)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # drop duplicates with same campaign id on date -> Unique campaign id and date are necessary
    # decision based on highest impressions (thats the value that is first in the funnel, so easiest to be triggered)
    df.sort_values(by=["segments.date","campaign.id","campaign.advertising_channel_type", "campaign.bidding_strategy_type", "metrics.impressions"], ascending=False, inplace=True)
    df = df.groupby(["segments.date","campaign.id", "campaign.advertising_channel_type", "campaign.bidding_strategy_type"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    
    return df
    
def load_productperformance(engine):
    """
    Loads product performance
    Arguments:
        name, .csv file name to load
    """
    # load data
    df = import_data_AWSRDS(schema="googleads",table="custom_productperformance_scd",engine=engine)

    # select columns
    df = df[[
        'segments.product_item_id',
        'segments.date',
        'campaign.advertising_channel_type',
        'campaign.advertising_channel_sub_type',
        'campaign.bidding_strategy_type',
        'metrics.ctr',
        'metrics.clicks',
        'metrics.average_cpc',
        'metrics.conversions',
        'metrics.cost_micros',
        'metrics.impressions',
        'metrics.search_click_share',
        'metrics.cost_per_conversion',
        'metrics.value_per_conversion',
        'metrics.search_impression_share',
        'metrics.conversions_from_interactions_rate',
        'metrics.search_absol___top_impression_share'
    ]]
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # drop duplicates with same segments.product_item_id on date -> Unique product id and date are necessary
    # decision based on highest impressions (thats the value that is first in the funnel, so easiest to be triggered)
    df.sort_values(by=["segments.date","segments.product_item_id","campaign.advertising_channel_type", "campaign.advertising_channel_sub_type", "campaign.bidding_strategy_type", "metrics.impressions"], ascending=False, inplace=True)
    df = df.groupby(["segments.date","segments.product_item_id", "campaign.advertising_channel_type", "campaign.advertising_channel_sub_type", "campaign.bidding_strategy_type"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)

    return df

def load_shopify_sceleton():
    """
    Loads shopify productsxdays_sceleton
    """
    
    table_name = "transformed"
    productsxdays_sceleton = pd.DataFrame(engine.execute("SELECT * FROM "+table_name+".shopify_productsxdays_sceleton").fetchall(), columns=engine.execute("SELECT * FROM "+table_name+".shopify_productsxdays_sceleton").keys())
    productsxdays_sceleton = productsxdays_sceleton[['daydate', 
                                                     'product_category', 
                                                     'product_category_number', 
                                                     #'variant_sku', 
                                                     'variant_id']]
        
    return productsxdays_sceleton


# UPDATED FOR MSC 
def load_sceleton(engine):
    """
    Loads shopify productsxdays_sceleton
    """
    
    table_name = "product"
    #productsxdays_sceleton = pd.DataFrame(engine.execute("SELECT * FROM "+table_name+".msc_custom_productsxdays_sceleton_v2").fetchall(), columns=engine.execute("SELECT * FROM "+table_name+".msc_custom_productsxdays_sceleton_v2").keys())
    
    productsxdays_sceleton = import_data_AWSRDS(schema=table_name,table="productsxdays_sceleton",engine=engine)
    productsxdays_sceleton = productsxdays_sceleton[[
        "nogl_id",
        'daydate',
        "shopify_variant_id",
        "shopify_variant_sku",
        "shopify_product_category_number",
        "shopify_product_category",
        "amazon_variant_asin",
        "amazon_variant_sku",
        "amazon_product_category",
        "amazon_product_category_number",
        
    ]]
        
    return productsxdays_sceleton


# ### Transformation

# In[7]:


def get_metric_list(df):
    """

    :param df: dataframe
    :return: list of metrics
    """

    metrics_list = list()
    for i in list(df.columns):
        if i.startswith("metrics"):
            metrics_list.append(i)
    print("Number of level 4 metrics: ", len(metrics_list))
    print("Level 4 metrics are:")
    i = 1
    for m in metrics_list:
        print("Metric ", i, "-> ", m)
        i =i+1

    return metrics_list


def create_dateframe(datacollection_startdate = "2020-01-01",days_into_future=120):
    """
    Creates a dataframe with one column "date" starting on datacollection_startdate
    until today plus days_into_future
    """

    end = date.today() + timedelta(days=days_into_future)

    start_date, end_date = datacollection_startdate, date(int(end.strftime("%Y")), int(end.strftime("%m")) , int(end.strftime("%d")))

    date_range = pd.date_range(start_date, end_date, freq='d')
    dateframe = pd.DataFrame(date_range, columns= ["date"]).sort_values("date",ascending=False)
    dateframe.reset_index(drop=True, inplace=True)

    return dateframe

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


# # 2. Product Performance
# 
# ### Decision on how to aggregate columns
# 
# sum: metric_sum variable in run_pp_pipeline function
# 
# mean: metric_mean variable in run_pp_pipeline function

# In[8]:


def add_level_1(df):
    """
    Adds level 1 based on 'campaign.advertising_channel_type'
    :param df: dataframe
    :return: dataframe
    """
    df["level1"] = df["campaign.advertising_channel_type"]
    df["level1"] = df["level1"].str.replace("_","-")
    return df

def add_level_2(df):
    """
    Adds level 2 based on 'campaign.bidding_strategy_type'
    :param df: dataframe
    :return: dataframe
    """
    
    df.reset_index(drop=True, inplace=True)
    df["level2"] = ""
    df["level2"] = df[df["level1"]=="SHOPPING"]["campaign.bidding_strategy_type"]
    df["level2"] = df["level2"].str.replace("_","-")

    return df

def add_col(df, column_name, metrics):
    """
    Given a list of metrics adds them in combination with a column_name.

    Arguments:
        - df, dataframe
        - column_name, string
        - metrics, list of string with the names of metrics
    """

    # for each metric
    for metric in metrics:

        # set new column name
        metric_name = metric.split(".")[1]
        col_name = column_name + "_" + metric_name
        levels = column_name.split("_")

        # set new column
        # ad metric from df_metric if condition is True else adds 0
        if len(levels)==1:
            df[col_name] = np.where((df["level1"] == levels[0]), df[metric], np.nan)
        if len(levels)==2:
            df[col_name] = np.where(((df["level1"] == levels[0]) & (df["level2"] == levels[1])), df[metric], np.nan)

    return df



def merge_metrics_with_levels(df):
    """
    Adds metrics to a dataframe given the existing columns.
    Arguments:
        - df, dataframe
        - metrics, a list of strings with the metrics to add to each level
    """

    level_1 = ["SHOPPING", "PERFORMANCE-MAX"]
    level_2 = ['TARGET-SPEND','MANUAL-CPC','MAXIMIZE-CONVERSION-VALUE', 'TARGET-ROAS']

    # get list of metrics
    metrics = [x for x in list(df.columns) if x.startswith("metrics.")]

    # level 1
    for lvl1_value in level_1:

        # merge level 1 with metrics
        df = add_col(df, lvl1_value, metrics)

        # do not continue if level 1 isn't SHOPPING
        if lvl1_value != "SHOPPING":
            continue

        # level 2
        for lvl2_value in level_2 :

            # set new column name
            column_name = lvl1_value + "_" + lvl2_value

            # merge level 2 with metrics
            df = add_col(df, column_name, metrics)


    return df

def clean_product_id(df):
    """
    Use the last number as the key id e.g. shopify_de_7078711984279_40698834387095 --> key id is 40698834387095

    :param df: dataframe
    :return: dataframe
    """
    df['segments.product_item_id'] = df['segments.product_item_id'].str.split('_').str[-1]
    df.dropna(subset=['segments.product_item_id'], inplace=True)

    return df

def drop_columns(df,columns):
    """
    Drops a list of columns from a dataframe

    Arguments:
        - df, dataframe
        - columns, a list of strings
    """


    df = df.drop(columns, axis=1)
    return df

def standardize_columns(df, first_cols=[]):
    """
    Orders the columns to guarantee a succesfull training.

    Arguments:
        - df, dataframe
        - first_cols, list of columns that will stay as first columns,
    """

    # order columns alphabetically
    df_columns = sorted(list(df.columns))
    order = [x for x in df_columns if x not in first_cols]

    first_cols.reverse()
    for col in first_cols:
        order.insert(0,col)

    df = df.loc[:, order]

    return df

def aggregate_by(df, metrics_sum, metrics_mean, index):
    """
    Aggregates metrics with values by date and product, this guarantees non-repeated dates and products in the dataframe.

    Arguments:
        - df, dataframe
        - metrics_sum, list of metric to aggregate by sum
        - metrics_mean, list of metric to aggregate by mean
        - index, list of columns to be used as

    """
    columns = list(df.columns)
    aggregate = {}
    default_value_if_no_values = 0

    # map an aggregate function to each metric
    for col in columns:
        # sum as agregate function
        for sum_col in metrics_sum:
            if col.endswith(sum_col):
                aggregate[col] = np.sum

        # mean as agregate function
        for mean_col in metrics_mean:
            if col.endswith(mean_col):
                aggregate[col] = np.mean

    # create pivot table with agregate functions selected
    # indexes are ga_date and ga_productsku
    df = pd.pivot_table(df, index=index, aggfunc=aggregate)
    df.reset_index(inplace=True)

    # to ensure feature consistency across clients we add the features that were dropped during the aggregation (e.g. 0 value mean columns)
    for col in list(aggregate):
            if col not in df.columns:
                df[col] = default_value_if_no_values

    return df

def fill_nan(df):
    """
    Fill nan values by 0.
    :param df: dataframe
    :return: dataframe
    """

    df.fillna(0,inplace=True)

    return df

def run_pp_pipeline(df):
    """
    Executes all the transformation of product performance

    :param df: dataframe with product performance
    :return: dataframe
    """
    t = Timer("Calculating level 1")
    df = add_level_1(df)
    t.end()

    t = Timer("Calculating level 2")
    df = add_level_2(df)
    t.end()

    t = Timer("Merging metrics with levels")
    df = merge_metrics_with_levels(df)
    t.end()

    t = Timer("Cleaning product_item_id")
    df = clean_product_id(df)
    t.end()

    # sort by date
    df = df.sort_values(by =['segments.date', 'segments.product_item_id'],ascending = [True, True])

    t = Timer("Drop columns")
    columns = [
        "campaign.advertising_channel_type",
        "campaign.advertising_channel_sub_type",
        "campaign.bidding_strategy_type",
        "metrics.ctr",
        "metrics.clicks",
        "metrics.average_cpc",
        "metrics.conversions",
        "metrics.cost_micros",
        "metrics.cost_micros",
        "metrics.impressions",
        "metrics.search_click_share",
        "metrics.cost_per_conversion",
        "metrics.value_per_conversion",
        "metrics.search_impression_share",
        "metrics.conversions_from_interactions_rate",
        "metrics.search_absol___top_impression_share",
        "level1",
        "level2"
    ]
    df = drop_columns(df, columns)
    t.end()

    t = Timer("Aggregation by date and product")
    metrics_sum=[
        "clicks",
        "conversions",
        "cost_micros",
        "impressions"
    ]
    metrics_mean=[
        "ctr",
        "average_cpc",
        "value_per_conversion",
        "cost_per_conversion",
        "search_click_share",
        "search_impression_share",
        "conversions_from_interactions_rate",
        "search_absol___top_impression_share"
    ]
    df = aggregate_by(df, metrics_sum, metrics_mean, index=["segments.date","segments.product_item_id"])
    t.end()

    t = Timer("Standardize columns order")
    standardize_columns(df, first_cols=["segments.date" ,"segments.product_item_id"])
    t.end()

    return df

def add_cp_level_1(df):
    """
    Adds level 1 based on campaign.advertising_channel_type column

    :param df: dataframe
    :return: dataframe
    """
    df["level1"] = df["campaign.advertising_channel_type"]
    df["level1"] = df["level1"].str.replace("_","-")
    return df


def merge_cp_metrics_with_levels(df):
    """
    Adds metrics to a dataframe given the existing columns.
    Arguments:
        - df, dataframe
        - metrics, a list of strings with the metrics to add to each level
    """

    level_1 = ['SHOPPING', 'SEARCH', 'DISCOVERY', 'DISPLAY', 'PERFORMANCE-MAX']

    # get list of metrics
    display_metrics = [x for x in list(df.columns) if x.startswith("metrics.")]
    metrics = [x for x in display_metrics if not "video" in x]

    # level 1
    for lvl1_value in level_1:

        # merge level 1 with metrics
        if lvl1_value=="DISPLAY":
            df = add_col(df, lvl1_value, display_metrics)
        else:
            df = add_col(df, lvl1_value, metrics)

    return df

def add_active_campaign(df):
    """
    Adds the column active_campaign.
    The campaign is considered active if the number of metrics.impressions is higher than 0

    :param df: dataframe
    :return: dataframe
    """
    df["active_campaign"] = np.where((df["metrics.impressions"] > 0),1,0)
    return df

def add_campaigns_per_day(df, values, channel):
    """
    Adds campaign per day based on values (columns to be added) of a channel

    :param df: dataframe
    :param values: a list of strings that refers to the column 'channel' values
    :param channel: the column to be analysed, for example 'campaign.advertising_channel_type'
    :return: dataframe
    """
    values = {x:[] for x in values}

    dates_values = list(df["segments.date"].tolist())
    campaign_id_values = list(df["campaign.id"].tolist())
    active_campaign = list(df["active_campaign"].tolist())
    channel = list(df[channel].tolist())

    current_date = None

    # for each date, ensure that we count campaign ids without repeating them by advertising value
    for i, date in enumerate(dates_values):
        # set list of campaign already seen for each advertising
        if date != current_date:
            campaigns = {x:[] for x in values}
        current_date = date

        # get row advertising value
        advertising = channel[i]

        # for each advertising value list add 1 if is the first time we see the campaign id for that advertising value, otherwise 0
        for key in values:
            if (key == advertising) and (not campaign_id_values[i] in campaigns[advertising]) and (active_campaign[i]==1):
                values[key].append(1)
                campaigns[advertising].append(campaign_id_values[i])
            else:
                values[key].append(0)


    # add columns
    for key in values:
        df[key+"_CAMPAIGNS_PER_DAY"] = values[key]

    return df

def add_campaigns_advertising_bidding_per_day(df):
    """
    Adds columns for campaigns per day based on advertising and bidding strategy

    :param df: dataframe
    :return: dataframe
    """

    # add advertising campaigns per day
    values = ['SHOPPING','SEARCH', 'DISCOVERY', 'DISPLAY', 'PERFORMANCE_MAX']
    df = add_campaigns_per_day(df,values,"campaign.advertising_channel_type")

    # add biding strategy campaigns per day
    values = ['MAXIMIZE_CONVERSION_VALUE', 'MANUAL_CPC', 'TARGET_SPEND', 'MAXIMIZE_CONVERSIONS', 'TARGET_CPA', 'TARGET_ROAS']
    df = add_campaigns_per_day(df,values,"campaign.bidding_strategy_type")

    return df

def run_cp_pipeline(df):
    """
    Executes all the transformations of campaign performance

    :param df: dataframe with campaign performance
    :return: dataframe
    """
    t = Timer("Calculating level 1")
    df = add_cp_level_1(df)
    t.end()

    t = Timer("Merging metrics with levels")
    df = merge_cp_metrics_with_levels(df)
    t.end()
    print("AFTER merge_cp_metrics_with_levels", filter_DF(df, [["segments.date","==","2021-08-05"],["campaign.advertising_channel_type","==","SEARCH"]])["metrics.average_cost"].mean())
    print("AFTER merge_cp_metrics_with_levels-OTHER", filter_DF(df, [["segments.date","==","2021-08-05"]])["SEARCH_average_cost"].mean())

    # sort by date
    df = df.sort_values(by =['segments.date'],ascending = [True])
    
    t = Timer("Add active campaign")
    df = add_active_campaign(df)
    t.end()
    print("AFTER add_active_campaign", filter_DF(df, [["segments.date","==","2021-08-05"],["campaign.advertising_channel_type","==","SEARCH"]])["metrics.average_cost"].mean())

    t = Timer("Add campaigns per day")
    df = add_campaigns_advertising_bidding_per_day(df)
    t.end()
    print("AFTER add_campaigns_advertising_bidding_per_day", filter_DF(df, [["segments.date","==","2021-08-05"],["campaign.advertising_channel_type","==","SEARCH"]])["metrics.average_cost"].mean())

    t = Timer("Drop columns")
    columns = [
        "level1",
        "metrics.content_impression_share",
        "metrics.view_through_conversions",
        "metrics.top_impression_percentage",
        "metrics.search_top_impression_share",
        "metrics.absolute_top_impression_percentage",
        "metrics.conversions_from_interactions_rate",
        "metrics.search_absol___top_impression_share",
        "metrics.interaction_rate",
        "metrics.conversions_value",
        "metrics.average_page_views",
        "metrics.search_click_share",
        "metrics.cost_per_conversion",
        "metrics.average_time_on_site",
        "metrics.percent_new_visitors",
        "metrics.value_per_conversion",
        "metrics.search_impression_share",
        "metrics.video_views",
        "metrics.video_view_rate",
        "metrics.video_quartile_p25_rate",
        "metrics.video_quartile_p50_rate",
        "metrics.video_quartile_p75_rate",
        "metrics.video_quartile_p100_rate",
        "metrics.engagements",
        "metrics.impressions",
        #"metrics.average_cost",
        "metrics.interactions",
        "metrics.relative_ctr",
        "metrics.engagement_rate",
        "campaign.id",
        "campaign.status",
        "campaign.start_date",
        "campaign.end_date",
        "campaign.advertising_channel_type",
        "campaign.bidding_strategy_type",
        "campaign_budget.amount_micros",
        "campaign_budget.reference_count",
        "metrics.ctr",
        "metrics.clicks",
        "metrics.average_cpc",
        "metrics.average_cpe",
        "metrics.average_cpm",
        "metrics.average_cpv",
        "metrics.bounce_rate",
        "metrics.conversions",
        "metrics.cost_micros"
    ]
    df = drop_columns(df, columns)
    t.end()
    print("AFTER drop_columns", filter_DF(df, [["segments.date","==","2021-08-05"]])["SEARCH_average_cost"].mean())

    t = Timer("Aggregation by date")
    metrics_sum=[
        "conversions_value", # because below there is already average conversion value
        "engagements",
        "impressions",
        "video_views",
        "interactions",
        "clicks",
        "conversions",
        "cost_micros",
        "view_through_conversions",
        "SHOPPING_CAMPAIGNS_PER_DAY",
        "SEARCH_CAMPAIGNS_PER_DAY",
        "DISCOVERY_CAMPAIGNS_PER_DAY",
        "DISPLAY_CAMPAIGNS_PER_DAY",
        "PERFORMANCE_MAX_CAMPAIGNS_PER_DAY",
        "MAXIMIZE_CONVERSION_VALUE_CAMPAIGNS_PER_DAY",
        "MANUAL_CPC_CAMPAIGNS_PER_DAY",
        "TARGET_SPEND_CAMPAIGNS_PER_DAY",
        "MAXIMIZE_CONVERSIONS_CAMPAIGNS_PER_DAY",
        "TARGET_CPA_CAMPAIGNS_PER_DAY",
        "TARGET_ROAS_CAMPAIGNS_PER_DAY"
    ]
    metrics_mean=[
        'relative_ctr',
        "ctr",
        "top_impression_percentage",
        "average_cost",
        "average_cpc",
        "average_cpe",
        "average_cpm",
        "average_cpv",
        "bounce_rate",
        "engagement_rate",
        "video_view_rate",
        "search_impression_share",
        "video_quartile_p25_rate",
        "video_quartile_p50_rate",
        "video_quartile_p75_rate",
        "video_quartile_p100_rate",
        "average_page_views",
        "search_click_share",
        "cost_per_conversion",
        "average_time_on_site",
        "percent_new_visitors",
        "value_per_conversion",
        "content_impression_share",
        "search_top_impression_share",
        "absolute_top_impression_percentage",
        "conversions_from_interactions_rate",
        "search_absol___top_impression_share",
        "interaction_rate"
    ]
    df = aggregate_by(df, metrics_sum, metrics_mean, index=["segments.date"])
    t.end()

    t = Timer("Standardize columns order")
    standardize_columns(df, first_cols=["segments.date"])
    t.end()

    t = Timer("Fill nan")
    df = fill_nan(df)
    t.end()

    return df

def merge_with_sceleton(df_pp,df_cp, engine, mergable_id):
    """
    Merges product performance and campaign performance after transformations with shopify sceleton

    :param df_pp: product performance dataframe
    :param df_cp: campaign performance dataframe
    :return: dataframe
    """

    # load shopify
    df_sceleton = load_sceleton(engine)
    #df_sceleton["daydate"]=df_sceleton["daydate"].apply(lambda x: x[0:11])
    df_sceleton["daydate"] = pd.to_datetime(df_sceleton["daydate"])
    df_sceleton[mergable_id] = df_sceleton[mergable_id].astype(str)
    print(df_sceleton.columns)

    # merge pp
    product_id = mergable_id
    df_pp.rename(columns={'segments.date': 'daydate','segments.product_item_id':product_id}, inplace=True)
    df_pp["daydate"] = pd.to_datetime(df_pp["daydate"])
    df_pp[mergable_id] = df_pp[mergable_id].astype(str)
    df_pp = df_pp.rename({col:'pp_'+ col for col in df_pp.columns[~df_pp.columns.isin(['daydate',product_id])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_pp,how='left', on=['daydate',product_id])


    # merge cp
    df_cp.rename(columns={'segments.date': 'daydate'}, inplace=True)
    df_cp["daydate"] = pd.to_datetime(df_cp["daydate"])
    df_cp = df_cp.rename({col:'cp_'+ col for col in df_cp.columns[~df_cp.columns.isin(['daydate'])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_cp,how='left', on='daydate')


    # nan as zero
    df_sceleton.fillna(0,inplace=True)

    return df_sceleton


def google_ads(client_name):
    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }
    
    best_mergable_id = get_best_mergable_id(client_name, "google_ads")
    df_cp_mapping = load_campaignperformance(engines["prod"], incl_name=True)
    sceleton = load_sceleton(engines["dev"])

    # get mapping of categories to campaigns
    df_cp_mapping["mapping"] = df_cp_mapping["campaign.name"].astype(str)

    # look for the columns that contains product category and get the unique
    unique_product_categories = []
    for col in sceleton.columns:
        if "product_category" in col:
            unique_product_categories.append(list(sceleton[col].unique()))
    unique_product_categories = list(set([item for sublist in unique_product_categories for item in sublist]))
    #unique_product_categories = list(sceleton.product_category.unique())
    unique_product_categories.append("Rest")

    for p in unique_product_categories:
        df_cp_mapping["mapping"] = df_cp_mapping["mapping"].apply(lambda x: str(p) if (str(p) in x) else x)
        
    df_cp_mapping["mapping"] = df_cp_mapping["mapping"].apply(lambda x: x if (x in unique_product_categories) else "")

    # rename "." to "_"
    for c in df_cp_mapping.columns:
        df_cp_mapping.rename(columns={c:c.replace(".","_")}, inplace=True)
        
    df_cp_mapping["mapping"] = df_cp_mapping["mapping"].replace("", "Rest")

    #df_cp_mapping.to_excel("Campaign_mapping_GoogleAds.xlsx")

    print("#"*20,"Product","#"*20)
    # product performance
    df_pp = load_productperformance(engines["prod"])
    df_pp = run_pp_pipeline(df_pp)

    print("\n"+"#"*20,"Campaign","#"*20)
    # campaign performance
    df_cp = load_campaignperformance(engines["prod"], incl_name=True)
    df_cp = run_cp_pipeline(df_cp)

    df_final = merge_with_sceleton(df_pp,df_cp, engines["dev"], mergable_id=best_mergable_id)

    print("\n"+"#"*20,"Test","#"*20)
    df_shopify = load_sceleton(engines["dev"])
    print("\nNumber of rows:")
    print("\tSceleton ORIGINAL:",len(df_shopify))
    #print("\tSceleton FINAL   :",len(df))

    print("\nNumber of columns:")
    print("\tSceleton ORIGINAL:",(df_shopify.shape[1]))
    print("\tProduct Performance:",(df_pp.shape[1]))
    print("\tCampaign Performance:",(df_cp.shape[1]))
    print("\tExpected FINAL:",(df_shopify.shape[1]+df_pp.shape[1]+df_cp.shape[1]-3)) # minus 3 because of dates from cp and pp, and product from pp
    print("\tSceleton FINAL:",(df_final.shape[1]))


    from sqlalchemy import create_engine, text
    # Assuming the rest of your code above this line is importing necessary modules and defining other variables...
    t = Timer("Export")
    # rename nogl id to variant_id
    df_final.rename(columns={"nogl_id":"variant_id"}, inplace=True)
    df_final.to_sql('msc_google_ads', con=engines["dev"], schema="transformed", if_exists='replace', index=False) # drops old table and creates new empty table
    t.end()

if __name__ == "__main__":
    google_ads(args.client_name)