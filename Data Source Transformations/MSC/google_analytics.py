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


def names_to_camelcase(names):
    """
    from a list of strings returns a dictionary 
    with the key being each name and the value the 
    correspondent name with camelCase, ignoring characters such as parenthesis and quotation marks
    """
    
    result = {}
    replace_chars = ["\"", "(", ")"]
    
    for name in names:
        # replace chars
        clean = name
        for char in replace_chars:
            clean = clean.replace(char,"")
            
        # spaces to camelcase
        clean = clean.split()
        clean = [x.capitalize() for x in clean]
        clean[0] = clean[0].lower()
        clean = "".join(clean)
        
        result[name] = clean
        
        
    return result



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
        
def remove_duplicates(df):
    """
    Deletes duplicate rows, the rows must be exactly the same
    """
    return df.drop_duplicates()


def load_main(engine,name=None):
    """
    Loads Google Analytics for Main page performance data from DB
    """
    
    table_name = "googleanalytics"
    df = import_data_AWSRDS(schema="googleanalytics",table="custom__report__airbyte__main_scd",engine=engine)
    
    df = df[['ga_date', 
                       'ga_medium', # dont use it!
                       'ga_source', 
                       'ga_campaign',
                       'ga_channelgrouping', # until here grouping levels
                       'ga_users', # from here performance data metrics
                       'ga_newusers',
                       'ga_sessions',
                       'ga_pageviews',
                       'ga_bouncerate',
                       'ga_avgtimeonpage',
                       'ga_sessionsperuser',
                       'ga_uniquepageviews', 
                       'ga_avgsessionduration',
                       'ga_pageviewspersession']]
    
    # remove duplicate rows
    df = remove_duplicates(df)
    
    # change datetime to str
    df["ga_date"] = df["ga_date"].apply(lambda x: date(int(x.strftime("%Y")), int(x.strftime("%m")) , int(x.strftime("%d"))))
    df = df.astype({"ga_date":"str"})
    
    return df



def load_pdp(engine, name=None):
    """
    Loads Google Analytics for product detail page (PDP) performance data from DB

    """
    
    table_name = "googleanalytics"
    df = import_data_AWSRDS(schema="googleanalytics",table="custom__report__airbyte_pdp_scd",engine=engine)
    
    df = df[['ga_date',
                  'ga_productsku',
                  'ga_productvariant',
                  'ga_source',
                  'ga_campaign',
                  'ga_channelgrouping',
                  'ga_productcategorylevel1', # until here grouping levels --> ignore product category for now!
                  'ga_itemrevenue', # from here performance data metrics
                  'ga_buytodetailrate',
                  'ga_carttodetailrate',
                  'ga_itemsperpurchase',
                  'ga_productcheckouts',
                  'ga_productaddstocart',
                  'ga_productdetailviews',
                  'ga_quantitycheckedout',
                  'ga_quantityaddedtocart',
                  'ga_productremovesfromcart']]
    
    # remove duplicate rows
    df = remove_duplicates(df)
    
    # change datetime to str
    df["ga_date"] = df["ga_date"].apply(lambda x: date(int(x.strftime("%Y")), int(x.strftime("%m")) , int(x.strftime("%d"))))
    df = df.astype({"ga_date":"str",
                    "ga_productsku":"str",
                    'ga_itemrevenue':"float", # from here performance data metrics
                    'ga_buytodetailrate':"float",
                    'ga_carttodetailrate':"float",
                    'ga_itemsperpurchase':"float",
                    'ga_productcheckouts':"float",
                    'ga_productaddstocart':"float",
                    'ga_productdetailviews':"float",
                    'ga_quantitycheckedout':"float",
                    'ga_quantityaddedtocart':"float",
                    'ga_productremovesfromcart':"float"})
    
    return df

def load_shopify_sceleton():
    """
    Loads shopify productsxdays_sceleton
    """
    
    table_name = "transformed"
    productsxdays_sceleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)
    productsxdays_sceleton = productsxdays_sceleton[['daydate', 
                                                     'product_category', 
                                                     'product_category_number', 
                                                     'variant_sku', 
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



def channel_replace(df, names, channel):
    """
    replaces ga_channelgrouping by a given channel 
    if ga_source is in a list of names
    """
    df.loc[ (df["ga_source"].str.lower().isin(names)) & (df["ga_channelgrouping"] == "(Other)"), "ga_channelgrouping"] = channel

def check_ig_fb_campaign_id(ga_source):
    """
    returns social if ga_source is an ig or fb campaign id
    Ids examples: 23850060441790039, '23849908589560039', '23849908589980039'
    """
    
    if (len(ga_source)==17 and ga_source.startswith("23")) or ga_source.startswith("ig_campaign_id"):
        return "social"
    
    return ga_source
    

def clean_level_2_other(df):
    """
    level 2 will be defined based on the value of ga_source 
    """
    
    # Social
    # ig or fb campaign ids on ga_source to social
    df["ga_source"] = df["ga_source"].apply(lambda x: check_ig_fb_campaign_id(x))
    
    # ig is assumed to be instagram and fb facebook
    names_to_social = ["ig","instagram","fb","facebook","facebook_ads","fb-group","pinterest","social"]
    channel_replace(df, names_to_social, "Social")
    
    # Organic Search
    names_to_organicsearch = ["google","adwords"]
    channel_replace(df, names_to_organicsearch, "Paid Search")

    # Email
    names_to_email = ["email","klaviyo"]
    channel_replace(df, names_to_email, "Email")

    # Referral
    names_to_referral = ["referral"]
    channel_replace(df, names_to_referral, "Referral")

    return df


def add_level_2(df):
    """
    adds the column level to df based on column ga_channelgrouping
    for rows where ga_channelgrouping equal to "(Other)" this functions 
    analysis ga_source to extract information about the channel and replace Other by the correct channel
    """
    # filter other
    df = clean_level_2_other(df)
    
    # clean column names from ga_channelgrouping to camelCase
    level2_names = list(df["ga_channelgrouping"].unique())
    level2_names_cleaned = names_to_camelcase(level2_names)
    
    df["level2"] = df["ga_channelgrouping"]
    df['level2'] = df["level2"].apply(lambda x: level2_names_cleaned[x])
    return df
    


# <br><br><br><br>
# ### 2.3.2 Level 3
# 
# - ga_medium was not considered given Tobi recommendation
# 
# - paidSearch had too many shoppingremarketingcatchall it was considered smartshopping 
# 
# - standarddisplay was considered as display
# 
# - performancemax and performancemaxtest was considered as pmax
# 
# - brand serach was considered search

# In[10]:


def level_3_social(name):
    """
    Given a name returns the following level3 for social level:
        - instagram
        - facebook
        - pinterest
        - youtube
        - tiktok
        - rest, none of above
        
    For the analysis is considered if the social network is in the name such as pinterest in in www.pinterest.com 
    and diminutives such as ig and fb
    
    Arguments: 
        - name, it represents ga_source
    """
    # convert to lowercase
    name = name.lower()
    
    # check if name contains any major social network
    names = ["instagram","facebook","pinterest","youtube","snapchat","tiktok"]
    for x in names:
        if x in name:
            return x
    
    # search for exact names
    # instagram
    ig_exact = ["ig", "igshopping"]
    if name in ig_exact:
        return "instagram"
    
    # facebook
    fb_exact = ["fb"]
    if name in fb_exact:
        return "facebook"
    
    fb_contains = ["fb-group"]
    for item in fb_contains:
        if item in name:
            return "facebook"
    
    # default is rest
    return "rest"

def level_3_organicsearch(name):
    """
    Given a name returns the following level3 for organic search:
        - google
        - rest
    
    Arguments: 
        - name, it represents ga_source
    """
    
    if name == "google":
        return name
    
    return "rest"

def level_3_display(name):
    """
    Given a name returns the following level3 for display:
        - google
        - rest
    
    Arguments: 
        - name, it represents ga_source
    """
    
    if name == "google":
        return name
    
    return "rest"


def level_3_paidsearch(campaign):
    """
    Given a name returns the following level3 for paid search:
        - search
        - smartshopping
        - discovery
        - pmax
        - display
        - rest, none of the above
    
    Arguments: 
        - campaign, it represents ga_campaign
    """
   
    
    # remove spaces and convert to lowercase
    name = campaign.lstrip().strip().lower().replace(" ","")
    
    allowed_names = {
        "smartshopping":["smartshopping","shoppingremarketingcatchall","smart"],
        "discovery":["discovery"],
        "pmax":["pmax","performancemax"],
        "display":["display"],
        "search":["brandsearch","brandedsearch","search"],
    }
   
    # for each level
    for level3 in allowed_names:
        # for each name of allowed names
        for value in allowed_names[level3]:
            # check if value in campaign, if so return corresponding level 3
            if value in name:
                return level3

   
    
    return "rest"

def lvl3_by_lvl2(level2, ga_source, ga_campaign):
    """
    Returns the level 3 value given the values of  'level2', 'ga_source' and 'ga_campaign'.
   
    Arguments:
        - level2, a string
        - ga_source, a string
        - ga_campaign, a string
    """
    if level2 == "social":
        return level_3_social(ga_source)
    
    if level2 == "organicSearch":
        return level_3_organicsearch(ga_source)
    
    if level2 == "paidSearch":
        return level_3_paidsearch(ga_campaign)

    if level2 == "display":
        return level_3_display(ga_source)
    
    return ""
    
def add_level_3(df):
    """
    Adds the column 'level3' to the dataframe considering the column 'level2' 
    and given level 2 value the columns 'ga_source' and 'ga_campaign'.
    
    Arguments:
        - df, a pandas dataframe with the columns 'level2', 'ga_source' and 'ga_campaign'
    """

    df["level3"] = df.apply(lambda x: lvl3_by_lvl2(x["level2"], x["ga_source"], x["ga_campaign"]),axis=1)
    return df


# ### 2.3.3 Level 4

# In[11]:


def social_level(ga_campaign):
    """
    Based on ga_campaign returns 'paid' or 'organic'
    
    Arguments:
        - ga_campaign, a string
    """
    
    if ga_campaign == "(not set)":
        return "organic"
    
    return "paid"
    

def add_level_4(df):
    """
    Adds 'level4' column to a dataframe.
    Adds an empty string for level 2 != 'social'.
    
    Arguments:
        - df, a pandas dataframe with the columns 'level2', 'level3' and 'ga_campaign'
    """
    
    df.reset_index(drop=True, inplace=True)
    # add level 4 value
    df["level4"] = df[df["level2"]=="social"].apply(lambda x: social_level(x["ga_campaign"]),axis=1)

    # default value is empty
    df["level4"] = df["level4"].fillna("")
     
    return df



def null_values_counter(df):
    """
    Prints the number of null values
    Arguments:
        - df, dataframe
    """
    
    null_values =  df.isnull().sum().sum()
    print("Null values:",null_values)
    
def add_col(df, column_name, metrics):
    """
    Given a list of metrics adds them in combination with a column_name 
    
    Arguments:
        - df, dataframe
        - column_name, string
        - metrics, list of string with the names of metrics
    """
    
    # for each metric
    for metric in metrics:
        
        # set new column name
        metric_name = metric.split("_")[1]
        col_name = column_name + "_" + metric_name
        levels = column_name.split("_")
        
        # set new column
        # by default is metric
        if len(levels)==1:
            df[col_name] = np.where((df["level2"] == levels[0]),df[metric], np.nan)
        if len(levels)==2:
            df[col_name] = np.where(((df["level2"] == levels[0]) & (df["level3"] == levels[1])),df[metric], np.nan)
        if len(levels)==3:
            df[col_name] = np.where(((df["level2"] == levels[0]) & (df["level3"] == levels[1]) & (df["level4"] == levels[2])),df[metric], np.nan)
        
        
    return df

def standardize_columns(df, metrics):
    """
    Orders the columns to guarantee a succesfull training.
    
    Arguments:
        - df, dataframe
        - metrics, list of strings, 
    """
    
    # order columns alphabetically, but ga_date keeps the first position and ga_productsku second if exists 
    df_columns = sorted(list(df.columns))
    order = [x for x in df_columns if x !="ga_date"]
    
    if "ga_productsku" in df_columns:
        df_columns.remove("ga_productsku")
        df.columns.insert(0,"ga_productsku")
    
    order.insert(0,"ga_date")
    df = df.loc[:, order]
    
    return df
                 
def merge_metrics_with_levels(df, metrics):
    """
    Adds metrics to a dataframe given the existing columns.
    Arguments:
        - df, dataframe
        - metrics, a list of strings with the metrics to add to each level
    """
    level_2 = ['organicSearch','paidSearch','display', 'social', 'referral', 'direct', 'email', 'other']
    level_3 = {
        "social" : ["facebook","instagram","pinterest","tiktok","youtube","snapchat","rest"],
        "paidSearch": ["search","smartshopping","display","discovery","pmax","rest"],
        "organicSearch":  ["google","rest"],
        "display": ["google","rest"]
    }
    level_4 = ["paid","organic"]
    
    # level 2
    for lvl2_value in level_2:
        
        df = add_col(df, lvl2_value, metrics) 

        if not lvl2_value in level_3:
            continue
            
        # level 3
        for lvl3_value in level_3[lvl2_value] :
            
            column_name_3 = lvl2_value + "_" + lvl3_value
            df = add_col(df, column_name_3, metrics)
            
            if not lvl2_value =="social":
                continue
                
            # level 4
            for lvl4_value in level_4:
                
                column_name_4 = column_name_3 + "_" + lvl4_value
                df = add_col(df, column_name_4, metrics)
    
    
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

def aggregate_by_date(df, metrics): 
    """
    Aggregates metrics with values by date, this guarantees non repeated dates in the dataframe.
    
    Arguments:
        - df, dataframe
        - metrics, a list of strings with the metrics to add to each level
   
    """
   
    metrics_sum = ['users', 'newusers', 'sessions','pageviews', 'uniquepageviews']
    metrics_mean = ['bouncerate','avgtimeonpage','sessionsperuser','avgsessionduration','pageviewspersession']
    
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
    
    # create pivot table with index ga_date
    df = pd.pivot_table(df, index=['ga_date'], aggfunc=aggregate)
    df.reset_index(inplace=True)

    # to ensure feature consistency across clients we add the features that were dropped during the aggregation (e.g. 0 value mean columns)
    for col in list(aggregate):
            if col not in df.columns:
                df[col] = default_value_if_no_values
    
    # standardization of columns
    df = standardize_columns(df, metrics)
    
    return df

def aggregate_by_date_product(df, metrics):
    """
    Aggregates metrics with values by date and product, this guarantees non repeated dates and products in the dataframe.
    
    Arguments:
        - df, dataframe
        - metrics, a list of strings with the metrics to add to each level
   
    """
    metrics_sum = ['itemrevenue','productcheckouts',
                   'productdetailviews','quantitycheckedout','productaddstocart',
                   'quantityaddedtocart','productremovesfromcart']
    metrics_mean = ['buytodetailrate','carttodetailrate','itemsperpurchase']
    
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
    df = pd.pivot_table(df, index=['ga_date', 'ga_productsku'], aggfunc=aggregate)
    df.reset_index(inplace=True)

    # to ensure feature consistency across clients we add the features that were dropped during the aggregation (e.g. 0 value mean columns)
    for col in list(aggregate):
            if col not in df.columns:
                df[col] = default_value_if_no_values
    
    # standardization of columns
    df = standardize_columns(df, metrics)
  
    return df

def basic_pipeline(df):
    """
    This pipeline adds levels 2, 3 and 4.
    Is used by both datasets main and pdp
    Arguments:
        - df, dataframe
    """
    
    # add level 2
    t = Timer("Calculating level 2")
    df = add_level_2(df)
    t.end()
    
    # add level 3
    t = Timer("Calculating level 3")
    df = add_level_3(df)
    t.end()
    
    # add level 4
    t = Timer("Calculating level 4")
    df = add_level_4(df)
    t.end()
    
    return df
        
def run_main_pipeline(df, remove_duplicates_activated=True):
    """
    Executes all methods to produce the final dataset 
    for main given the original dataframe from airbyte.
    Removes duplicated rows.
    
    Arguments:
        - df, dataframe
    """
    
    df = basic_pipeline(df)
    
    # remove duplicates
    if remove_duplicates_activated:
        t = Timer("Remove duplicates")
        df = remove_duplicates(df)
        t.end()
    
    # add columns
    t = Timer("Merging metrics with levels")
    metrics = [
        'ga_users', 'ga_newusers', 'ga_sessions',
        'ga_pageviews', 'ga_bouncerate', 'ga_avgtimeonpage',
        'ga_sessionsperuser', 'ga_uniquepageviews', 'ga_avgsessionduration',
        'ga_pageviewspersession'
    ]
    df = merge_metrics_with_levels(df,metrics)
    t.end()
       
    # drop columns
    t = Timer("Dropping columns")
    columns = ['ga_users','ga_source','ga_campaign','ga_channelgrouping', 'ga_newusers', 'ga_sessions',
                'ga_pageviews', 'ga_bouncerate', 'ga_avgtimeonpage',
                'ga_sessionsperuser', 'ga_uniquepageviews', 'ga_avgsessionduration',
                'ga_pageviewspersession','level2','level3','level4','ga_medium']
    df = drop_columns(df,columns)
    t.end()
    
    # agregate by date
    t = Timer("Aggregation by date")
    df = aggregate_by_date(df,metrics)
    t.end()
    
    df = df.sort_values(by = 'ga_date')
  
    return df

def run_pdp_pipeline(df, remove_duplicates_activated=True):
    """
    Executes all methods to produce the final dataset 
    for pdpd given the original dataframe form airbyte.
    Removes duplicated rows.
    
    Arguments:
        - df, dataframe
    """
    
    df = basic_pipeline(df)
    
    # remove duplicates
    if remove_duplicates_activated:
        t = Timer("Remove duplicates")
        df = remove_duplicates(df)
        t.end()
    
    # add columns
    t = Timer("Merging metrics with levels")
    metrics = [
        'ga_itemrevenue', 'ga_buytodetailrate','ga_carttodetailrate',
        'ga_itemsperpurchase', 'ga_productcheckouts','ga_productaddstocart',
        'ga_productdetailviews','ga_quantitycheckedout','ga_quantityaddedtocart',
        'ga_productremovesfromcart'
    ]
    df = merge_metrics_with_levels(df, metrics)
    t.end()

    # drop columns
    t = Timer("Dropping columns")
    columns = ['ga_productvariant','ga_source','ga_campaign','ga_channelgrouping',
                'ga_productcategorylevel1','ga_itemrevenue', 'ga_buytodetailrate','ga_carttodetailrate',
                'ga_itemsperpurchase','ga_productcheckouts','ga_productaddstocart','ga_productdetailviews',
                'ga_quantitycheckedout','ga_quantityaddedtocart','ga_productremovesfromcart','level2','level3','level4'] 
    df = drop_columns(df,columns)
    t.end()
    
    # agregate by date
    t = Timer("Aggregation by date and product")
    df = aggregate_by_date_product(df, metrics)
    t.end()
    
    df = df.sort_values(by =['ga_date', 'ga_productsku'],ascending = [True, True])

    return df


# ## 2.5 Consolidation

# In[13]:


def bfill(df,col):
    """
    Backward fills a list
    Fill zeros and NaN values
    returns the list
    """
    series = df[col].replace(0,pd.np.nan).bfill()  
    return series
            
def ffill(df,col):
    """
    Forward fills a list
    Fill zeros and NaN values
    returns the list
    """
    series = df[col].replace(0,pd.np.nan).ffill()
    return series
    
   
    
def merge_dates(df):
    """
    Adds dates from 2020-01-01 until last date plus 120.
    Adds columns as np.nan if the date doesnt exist in df
    """
    
    df_dates = create_dateframe(datacollection_startdate = "2020-01-01",days_into_future=120)
    
    # reverse to chrono order
    df_dates=df_dates.iloc[::-1]
    
    # get cols
    df_cols = list(df.columns)
    data = {x:[] for x in df_cols if x != "ga_date"}
    data["date"] = []
    
    
    row_idx = 0
    df_rows = df.to_dict()
    len_df = df.shape[0]
  
    # for each date
    for date in df_dates["date"].astype(str):
        
        # add date
        data["date"].append(date)
        
        
        # if date not in dataframe add nan to all columns
        if row_idx == len_df or date < df_rows["ga_date"][row_idx]:
            for col in df_cols:
                if col == "ga_date":
                    continue
                data[col].append(np.nan)
        elif date == df_rows["ga_date"][row_idx]:
            for col in df_cols:
                if col == "ga_date":
                    continue
                data[col].append(df[col][row_idx])
            row_idx+=1
        else:
            print(date)
        
    df = pd.DataFrame(data)
    
    return df

def merge_pdp_dates(df):
    """
    Adds dates from 2020-01-01 until last date plus 120
    for PDP, this function will create a date for each product


    Adds columns as np.nan if the date doesnt exist in df
    """
    
    df_dates = create_dateframe(datacollection_startdate = "2020-01-01",days_into_future=120)
    # reverse to chrono order
    df_dates=df_dates.iloc[::-1]
    
    # get cols
    df_cols = list(df.columns)
    data = {x:[] for x in df_cols if x != "ga_date"}
    data["date"] = []
    
    
    row_idx = 0
    df_rows = df.to_dict()
    len_df = df.shape[0]
    
    unique_products = df["ga_productsku"].unique()
    unique_products.sort()
  
    # for each date
    for date in df_dates["date"].astype(str):
        
        # add date
        for product in unique_products:
            data["date"].append(date)
            data["ga_productsku"].append(product)
        
        
            # if date not in dataframe add nan to all columns
            if row_idx == len_df or date < df_rows["ga_date"][row_idx] or (date == df_rows["ga_date"][row_idx] and product < df_rows["ga_productsku"][row_idx]):
                for col in df_cols:
                    if col in ["ga_date","ga_productsku"]:
                        continue
                    data[col].append(np.nan)
                    
            elif date == df_rows["ga_date"][row_idx] and product == df_rows["ga_productsku"][row_idx]:
                for col in df_cols:
                    if col in ["ga_date","ga_productsku"]:
                        continue
                    data[col].append(df[col][row_idx])
                    
                row_idx+=1
            else:
                print(date)
                
    for col in data:
        print(col,len(data[col]))
    df = pd.DataFrame(data)
    
    return df


def analyse_nan_values(df, product=False): # something is wrong with the while loops
    """
    Prints an analysis about the NaN values.
    It shows the number of rows with at least one NaN.
    """
    if product:
        date_col="ga_date"
    else:
        date_col="date"
    
    # get number of total rows
    n_rows = df.shape[0]
    
    nan_df = df.isnull().any(axis=1)
    dates_nan = list(df[nan_df][date_col].tolist())
    
    consecutive = []
    for i in range(len(dates_nan)-1):
        start_date = dates_nan[i]
        consecutive_days = 0
        
        d1 = datetime.strptime(dates_nan[i], "%Y-%m-%d")
        d2 = datetime.strptime(dates_nan[i+1], "%Y-%m-%d")

        # difference between dates in timedelta
        delta = d2 - d1
        days = delta.days

        consecutive.append(days)
          
        
    # beginning
    print()
    print("-"*20,"NaN Analysis","-"*20)
    print(df.iloc[0][date_col], "- first date")
    
    # get number of nan beginning
    first_idx = 0
    date = ""
    counter = 0
    
    print(df)
    
    while consecutive[first_idx] == 1:
        date = dates_nan[first_idx]
        first_idx += 1
    
    print("   ",first_idx+1,"consecutive nan rows")
    print(date)
    
    # get number of nan from end of real data until the future
    last_idx = len(consecutive)-1
    counter = 0
    while consecutive[last_idx] == 1:
        counter += 1
        date = dates_nan[last_idx]
        last_idx -= 1
    
    n_nan = sum(consecutive[first_idx:last_idx])/n_rows*100
    print("   ",round(n_nan,1),"% nan")
    
    # last date
    print(date)
    print("   ",counter,"consecutive nan rows")
    print(df.iloc[-1][date_col], "- last date")
    print("-"*50)
    
    
def fill_nan(df, ffill_cols=[], bfill_cols=[], product=False):
    """
    Receiveis a list of columns and backwards fills and then forwards fills.
    Done for series sorted by ascending date.
    Then fill all nan values with 0.
    
    Arguments:
        - df, dataframe
        - bf_fill_cols, backward and forward fill columns
    """
    #if not product:
        #analyse_nan_values(df, product)
    
    df_cols = list(df.columns)
    fcols = []
    bcols = []
    for df_col in df_cols:
        # for each column check if endswith one of ffill_cols
        for col in ffill_cols:
            if df_col.endswith(col):
                fcols.append(df_col)
                continue
        # for each column check if endswith one of bfill_cols
        for col in bfill_cols:
            if df_col.endswith(col):
                bcols.append(df_col)
                continue
                
    for col in fcols:
        df[col] = ffill(df,col,product)
    
    for col in bcols:
        df[col] = bfill(df,col,product)
    
    # fill with 0 remainer nan values
    df.fillna(0,inplace=True)
    
    return df

def merge_with_sceleton(df_main,df_pdp, engine, mergable_id):
    """
    merge main and product into shopify sceleton
    returns a dataframe with the same number of rows than shopify sceleton
    """
    # load shopify
    df_sceleton = load_sceleton(engine)
    #df_sceleton["daydate"]=df_sceleton["daydate"].apply(lambda x: x[0:11])
    df_sceleton["daydate"] = pd.to_datetime(df_sceleton["daydate"])
    df_sceleton[mergable_id] = df_sceleton[mergable_id].astype(str)
    
    # merge main 
    df_main.rename(columns={'date': 'daydate'}, inplace=True)
    df_main["daydate"] = pd.to_datetime(df_main["daydate"])
    df_main = df_main.rename({col:'main_'+ col for col in df_main.columns[~df_main.columns.isin(['daydate'])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_main,how='left', on='daydate')
    
    # merge pdp
    product_id = mergable_id
    df_pdp.rename(columns={'ga_date': 'daydate','ga_productsku':product_id}, inplace=True)
    df_pdp["daydate"] = pd.to_datetime(df_pdp["daydate"])
    df_pdp = df_pdp.rename({col:'pdp_'+ col for col in df_pdp.columns[~df_pdp.columns.isin(['daydate',product_id])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_pdp,how='left', on=['daydate',product_id])
    
    df_sceleton.fillna(0, inplace=True)
    
    return df_sceleton


def google_analytics(client_name):
    
    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }
    
    best_mergable_id = get_best_mergable_id(client_name, "google_analytics")

    df1 = load_main(engine=engines["prod"])
    df2 = load_pdp(engine=engines["prod"])
    """
    MAIN
    """
    df1 = run_main_pipeline(df1)
    t = Timer("Merge dates")
    df1 = merge_dates(df1)
    t.end()
    t = Timer("Fill nan")
    df1 = fill_nan(df1)
    t.end()

    """
    PDP
    """

    df2 = run_pdp_pipeline(df2)
    t = Timer("Merge dates")
    t.end()

    t = Timer("Fill nan")
    df2 = fill_nan(df2, product=True)
    t.end()

    t = Timer("Merge with sceleton")
    df_final = merge_with_sceleton(df1,df2, engines["dev"], mergable_id=best_mergable_id)
    t.end()


    from sqlalchemy import create_engine, text
    # Assuming the rest of your code above this line is importing necessary modules and defining other variables...
    t = Timer("Export")
    # rename nogl id to variant_id
    df_final.rename(columns={'nogl_id': 'variant_id'}, inplace=True)
    df_final.to_sql('msc_google_analytics', con=engines["dev"], schema="transformed", if_exists='replace', index=False) # drops old table and creates new empty table
    t.end()


if __name__ == "__main__":
    google_analytics(args.client_name)