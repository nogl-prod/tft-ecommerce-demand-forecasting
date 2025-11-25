#!/usr/bin/env python
# coding: utf-8

# # 1. Imports

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import time
import requests
from msal import ConfidentialClientApplication
from io import BytesIO


# In[2]:


import sys
sys.path.append("../")

# from support functions:



# # 2. Support Functions

# In[3]:


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


# ## 2.1 Transformation and Cleaning

# In[4]:

def load_campaignperformance(incl_name=False, prod_engine=None):
    """
    Loads campaign performance
    Arguments:
        name, .csv file name to load
    """
    # load data
    df = import_data_AWSRDS(schema="googleads",table="custom_campaignperformance_scd",engine=prod_engine)

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
    
def load_productperformance(prod_engine):
    """
    Loads product performance
    Arguments:
        name, .csv file name to load
    """
    # load data
    df = import_data_AWSRDS(schema="googleads",table="custom_productperformance_scd",engine=prod_engine)

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
# create empty date dataframe
def create_dateframe(days_into_future=120):
    
    end = date.today() + timedelta(days=days_into_future)

    start_date, end_date = datacollection_startdate, date(int(end.strftime("%Y")), int(end.strftime("%m")) , int(end.strftime("%d")))

    date_range = pd.date_range(start_date, end_date, freq='d')
    dateframe = pd.DataFrame(date_range, columns= ["daydate"]).sort_values("daydate",ascending=False)
    dateframe.reset_index(drop=True, inplace=True)
    return dateframe


# ### 2.1.1 Klaviyo

# #### Data Load Functions

# In[5]:


def load_klaviyo_data(DATASOURCE_FOLDER, filename):
    data = pd.read_csv(DATASOURCE_FOLDER+filename)
    return data


# In[6]:


def load_data_from_klaviyo_json(url):
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    json_data = json.loads(response.text)
    data = json_data["data"]
    data = pd.json_normalize(data, max_level=0)
    # print(json_data)
    return data, str(json_data["next"])


# #### Events raw data transformation

# In[7]:


def split_event_data(raw_events_df):
    '''
    Takes the raw klaviyo events dataframe and splits it into different types of events
    Return a dictionary containing seperate dataframes for checkout_started, ordered_product, cancelled_order, refunded_order and non_sku_events
    '''
    
    # Split SKU events by type, need to be handled seperately because events_properties json is not always build the same
    klaviyo_events_sku_checkoutStarted = raw_events_df[raw_events_df["event_name"] == "Checkout Started"]
    klaviyo_events_sku_orderedProduct = raw_events_df[raw_events_df["event_name"] == "Ordered Product"]
    klaviyo_events_sku_cancelledOrder = raw_events_df[raw_events_df["event_name"] == "Cancelled Order"]
    klaviyo_events_sku_refundedOrder = raw_events_df[raw_events_df["event_name"] == "Refunded Order"] 
    
    # Create non SKU event dataframe
    sku_list = ["Checkout Started", "Ordered Product", "Cancelled Order", "Refunded Order"]
    klaviyo_events_nonsku = raw_events_df[~raw_events_df["event_name"].isin(sku_list)]
    
    return {'checkout_started' : klaviyo_events_sku_checkoutStarted, 
            'ordered_product' : klaviyo_events_sku_orderedProduct, 
            'cancelled_order' : klaviyo_events_sku_cancelledOrder, 
            'refunded_order' : klaviyo_events_sku_refundedOrder, 
            'non_sku_events' : klaviyo_events_nonsku}


# #### SKU Event Preprocessing

# In[8]:


def klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(df, columnname, single_quotation_mark = True):
    """
    Takes the sku event df and the name of the final column and returns a df with event time, event type, sku id and number of units affected
    Used for the event types Checkout Started, Cancelled Order, Refunded Order
    Set single_quotation_mark = False if the event properties column of the source data uses double quatation marks ( " instead of ' )
    """

    # start extracting relevant json information from event_properties
    df = df.reset_index(drop=True)
    
    # Select correct method for parsing event_properties based on data source
    if single_quotation_mark == False:
        df_norm = pd.concat([df, pd.json_normalize(df["event_properties"].apply(json.loads))], axis=1)
    else:
        df_norm = pd.concat([df, pd.json_normalize(df["event_properties"].apply(eval))], axis=1)
    
    df_norm = df_norm[["datetime", "event_name", "event_properties", "Items", "$extra.line_items"]]

    # count number of skus per order
    df_norm["number_of_skus"] = df_norm["Items"].apply(lambda x: len(x))

    # max number of skus
    max_lineitems = df_norm["number_of_skus"].max()
    df_norm['datetime'] = pd.to_datetime(df_norm['datetime'], utc=True).dt.date

    # loop over number of items and extract skus in each column
    for i in range(max_lineitems):
        df_norm = pd.concat([df_norm, pd.DataFrame(df_norm['$extra.line_items'].values.tolist()).iloc[:, [i]]], axis=1)
        df_norm.rename(columns = {i:"line_item"+str(i+1)}, inplace=True)
        to_concat = pd.json_normalize(df_norm["line_item"+str(i+1)]).rename(columns={"sku":"sku"+str(i+1)})
        df_norm = pd.concat([df_norm, to_concat["sku"+str(i+1)]], axis=1)
        df_norm.drop(columns=["line_item"+str(i+1)], inplace=True)

    # transform to seperate rows
    df_norm.drop(columns=["event_properties","Items","$extra.line_items","number_of_skus"], inplace = True) # drop unneccesary columns
    df_norm = pd.melt(df_norm, id_vars=["datetime","event_name"]).dropna(axis=0) # melt all sku columns to one row and drop rows where no sku_id
    df_norm.drop(columns="variable", inplace=True) # drop variable column
    df_norm = df_norm.groupby(df_norm.columns.tolist(),as_index=False).size() # groupby to aggregate duplicate rows (several events per day)
    df_norm = df_norm.loc[df_norm["value"] != ""] # drop all columns where sku_id has no value
    df_norm.rename(columns={"value":"sku_id", "size":columnname}, inplace=True) # rename value column

    return df_norm


def klaviyo_Extract_sku_id_from_orderedProduct(df, columnname, single_quotation_mark = True):
    """
    Takes the sku event df and the name of the final column and returns a df with event time, event type, sku id and number of units affected
    Used for Ordered Product events
    Set single_quotation_mark = False if the event properties column of the source data uses double quatation marks ( " instead of ' )
    """

    # start extracting relevant json information from event_properties
    df = df.reset_index(drop=True)
    
    # Select correct method for parsing event_properties based on data source
    if single_quotation_mark == False:
        df_norm = pd.concat([df, pd.json_normalize(df["event_properties"].apply(json.loads))], axis=1)
    else:
        df_norm = pd.concat([df, pd.json_normalize(df["event_properties"].apply(eval))], axis=1)
        
    df_norm['datetime'] = pd.to_datetime(df_norm['datetime'], utc=True).dt.date
    df_norm = df_norm[["datetime", "event_name", "event_properties", "SKU", "Quantity"]]

    df_norm.drop(columns=["event_properties"], inplace = True) # drop unneccesary columns
    df_norm.rename(columns={"SKU":"sku_id", "Quantity":columnname}, inplace=True) # rename value column

    return df_norm


# #### Non SKU Event Functions

# In[9]:


def preprocess_nonsku_events(df):
    """
    Takes the non-sku event df, creates the correct combination of lvl 1 and lvl 2 features and returns a df in the correct form for the pivot_nonsku_events function
    """
    
    lvl_2_features = ['Received Email', 'Active on Site', 'Opened Email', 'Unsubscribed', 'Clicked Email', 'Placed Order', 'Fulfilled Order', 'Bounced Email',                  'Subscribed to List', 'Fulfilled Partial Order', 'Marked Email as Spam', 'Unsubscribed from List', 'Updated Email Preferences', 'Dropped Email']
   
    df = df[df['event_name'].isin(lvl_2_features)]    # select only relevant event types

    conditions = [
        df['flow_id'].isnull() & ~df['campaign_id'].isnull(), # Flow ID is NA but camp ID is NOT NA
        ~df['flow_id'].isnull() & df['campaign_id'].isnull(), # Flow ID is NOT NA but camp ID is NA
        df['flow_id'].isnull() & df['campaign_id'].isnull()   # Flow ID and camp ID are both NA
    ]
    lvl_1_features = ['Campaign', 'Flow', 'Other']

    df['camp_or_flow'] = np.select(conditions, lvl_1_features, default='NAN')   # Assign each event to Campaign, Flow or Other based on conditions
    
    if 'NAN' in df['camp_or_flow'].unique():   # Return an error if camp/flow status is ambiguous
        return print('ERROR: Row contains both Campaign and Flow ID')
    
    df['datetime'] = pd.to_datetime(df['datetime']).dt.date     # drop hours and minutes from datetime
    df['camp_flow_event_name'] =  df['camp_or_flow'] + ' - ' + df['event_name']   # create new column with the combination of lvl1 qnd lvl2 features
    df = df.sort_values(['camp_flow_event_name'])
    df.drop(columns=["camp_or_flow", 'event_name'], inplace=True) 
    
    return df


def pivot_nonsku_events(df):
    """
    Takes the output from preprocess_nonsku_events and returns a df containing the number of times each event occured on a day
    """
    
    lvl_2_features = ['Received Email', 'Active on Site', 'Opened Email', 'Unsubscribed', 'Clicked Email', 'Placed Order', 'Fulfilled Order', 'Bounced Email',                  'Subscribed to List', 'Fulfilled Partial Order', 'Marked Email as Spam', 'Unsubscribed from List', 'Updated Email Preferences', 'Dropped Email']
    
    all_feature_columns = [f'Campaign - {lvl_2_feature}' for lvl_2_feature in lvl_2_features] +     [f'Flow - {lvl_2_feature}' for lvl_2_feature in lvl_2_features] +     [f'Other - {lvl_2_feature}' for lvl_2_feature in lvl_2_features] 
    
    df = pd.pivot_table(df, values='id', index='datetime', columns='camp_flow_event_name', aggfunc='count')  # pivot table
    
    df = df.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime']) # Convert index to date time
    
    df.fillna(0, inplace = True)   # fill missing values with 0

    for feature in all_feature_columns:   # add columns for missing lvl1/lvl2 feauture combinations
        if not feature in df.columns:
            df[feature] = 0

    return df


# #### Campaign Functions Preprocessing

# In[10]:


def calculate_camp_recipients(df):
    """
    Takes the Klaviyo campaigns dataframe and returns a dataframe with the number of actual recipients per day
    """
    
    recipient_df = df[df['status'] == 'sent']  # Filter only sent campaigns 

    recipient_df['sent_at'] = pd.to_datetime(recipient_df['sent_at'], utc=True).dt.date  # Truncate date to year-month-day only

    recipient_df = recipient_df.groupby('sent_at').sum('num_recipients')   #  sum recipients for each day

    recipient_df.reset_index(inplace = True)  
    recipient_df['sent_at'] = pd.to_datetime(recipient_df['sent_at']) # Convert index to date 
    recipient_df = recipient_df[['sent_at', 'num_recipients']]  # Make sure there are exactly two columns in df

    return recipient_df


def calculate_camp_per_day(df):
    """
    Takes the Klaviyo campaigns dataframe and returns the number of campaigns sent out each day
    """
    
    camp_df = df[df['status'] == 'sent']  # Filter only sent campaigns 

    camp_df['sent_at'] = pd.to_datetime(camp_df['sent_at'], utc=True).dt.date  # Truncate date to year-month-day only

    camp_df = camp_df.groupby('sent_at')['_airbyte_campaigns_hashid'].count()   #  count campaigns for each day

    camp_df = camp_df.reset_index()  
    camp_df['sent_at'] = pd.to_datetime(camp_df['sent_at']) # Convert index to date time
    camp_df.rename(columns={"sent_at":"sent_at", "_airbyte_campaigns_hashid":"campaigns_sent"}, inplace=True) # rename campaigns columns

    return camp_df


def calculate_planned_recipients(campaigns_df, lists_df, campaigns_lists_df):
    """
    Takes three Klaviyo dataframes (campaigns, lists and campaigns_lists) and returns the sum of the number of people who were in a list that are associated with each 
    day's active campaigns. 
    Sent campaigns use historical data, planned campaigns extrapolate based on the most recent available list data
    Note that lists are likley non exclusive (i.e. double counting occurs) and this number is usually much higher than the number of actual recipients
    """
    
    planned_recp_df = campaigns_lists_df
    planned_recp_df = campaigns_lists_df.merge(lists_df, on = 'id', how = "left")  # Merge campaign list on list to get most recent number of people in list
    planned_recp_df = planned_recp_df.merge(campaigns_df, on = '_airbyte_campaigns_hashid', how = 'left') # Merge campaign to get send_time data
    
    planned_recp_df = planned_recp_df[planned_recp_df['status'] != 'cancelled']  # Cancelled campaings are not relevant and dropped
    planned_recp_df = planned_recp_df[~planned_recp_df['send_time'].isna()]  # Campaigns without a planned send_time can't integrated into the dataframe
    
    planned_recp_df['send_time'] = pd.to_datetime(planned_recp_df['send_time'], utc=True).dt.date  # Truncate date to year-month-day only

    planned_recp_df['planned_recipients'] = np.where(planned_recp_df['sent_at'].isna(), planned_recp_df['person_count_y'], planned_recp_df['person_count_x']) 
    # For historical data planned recipients is the actual number of people in the corresponding lists at the time. (from campaigns_lists)
    # For future data planned recipients is the most recent number of people in the corresponding lists. (from lists)
    
    planned_recp_df = planned_recp_df.groupby('send_time').sum('planned_recipients')  # sum planned recipients per day
    
    planned_recp_df.reset_index(inplace = True)
    planned_recp_df['send_time'] = pd.to_datetime(planned_recp_df['send_time']) # Convert index to date time
    planned_recp_df = planned_recp_df[['send_time', 'planned_recipients']]  # Make sure only the relevant data in in the final dataset
    
    return planned_recp_df


# #### Data Merging Functions

# In[11]:


def create_dateframe(days_into_future=120, datacollection_startdate = '01-01-20'):
    '''
    Create empty date dataframe
    '''
    
    end = date.today() + timedelta(days=days_into_future)

    start_date, end_date = datacollection_startdate, date(int(end.strftime("%Y")), int(end.strftime("%m")) , int(end.strftime("%d")))

    date_range = pd.date_range(start_date, end_date, freq='d')
    dateframe = pd.DataFrame(date_range, columns= ["daydate"]).sort_values("daydate",ascending=False)
    dateframe.reset_index(drop=True, inplace=True)
    
    return dateframe


def preprocess_campaigns_and_nonsku_events (campaigns_df, lists_df, campaigns_lists_df, nonsku_df, datacollection_startdate = '01-01-20', days_into_future = 120, nonsku_preprocessed = False):
    """
    Takes the campaigns, lists, campaigns_lists and non-sku event data and returns a day-by-day dataframe from datacollection_startdate to today + days_into_future 
    Output inlcudes data on number of recipients, number of campaigns and number of planned recipients (people in lists) as well as data on non-sku events
    """

    datacollection_startdate = datacollection_startdate  
    master_df = create_dateframe(days_into_future, datacollection_startdate)  # Create empty dataframe

    # Create dfs for the individual data columns
    camp_recipients = calculate_camp_recipients(campaigns_df)
    campaigns_per_day = calculate_camp_per_day(campaigns_df)  
    planned_recipients = calculate_planned_recipients(campaigns_df, lists_df, campaigns_lists_df)   
    
    if nonsku_preprocessed == True:
        nonsku_events = nonsku_df
    else: 
        nonsku_events = nonsku_df.pipe(preprocess_nonsku_events).pipe(pivot_nonsku_events)

    # Merge Campaign Data
    master_df = master_df.merge(camp_recipients, how = 'left', left_on = 'daydate', right_on = 'sent_at')
    master_df = master_df.merge(campaigns_per_day, how = 'left', left_on = 'daydate', right_on = 'sent_at')
    master_df = master_df.merge(planned_recipients, how = 'left', left_on = 'daydate', right_on = 'send_time')
    
    master_df = master_df[['daydate', 'num_recipients', 'campaigns_sent', 'planned_recipients']] # Drop extra time columns
    
    master_df = master_df.merge(nonsku_events, how = 'left', left_on = 'daydate', right_on = 'datetime') # Merge non-sku events 
    
    master_df.drop(columns=['datetime'], inplace=True) # Drop extra time column
    master_df = master_df.fillna(0)  # fill NAs with 0s
    
    return master_df


def create_master_skeleton(checkout_started_df, 
                           cancelled_order_df, 
                           refunded_order_df, 
                           ordered_product_df, 
                           nonsku_event_and_campaign_df):
    '''
    Takes the preprocessed klaviyo dataframes generated by the SKU event extraction functions and the non-sku-event/campaign function 
    and returns a master dataframe with a seperate row for every day X sku combination.
    '''
    
    # Drop superfluous event_name columns
    checkoutStarted_skus_by_date = checkout_started_df.drop("event_name", axis = 1)
    cancelledOrder_skus_by_date = cancelled_order_df.drop("event_name", axis = 1)
    refundedOrder_skus_by_date = refunded_order_df.drop("event_name", axis = 1)
    orderedProduct_skus_by_date = ordered_product_df.drop("event_name", axis = 1)
    
    # Merge the SKU event dataframes on sku and datetime
    skuXdate_df = checkoutStarted_skus_by_date.merge(cancelledOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    skuXdate_df = skuXdate_df.merge(refundedOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    skuXdate_df = skuXdate_df.merge(orderedProduct_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    
    skuXdate_df = skuXdate_df.fillna(0) # Events that did not occur on a given day X sku combination are filled in with zero
    
    # prepare columns for merging
    skuXdate_df.rename(columns = {'datetime' : 'daydate'}, inplace = True) 
    skuXdate_df['daydate'] = pd.to_datetime(skuXdate_df['daydate'])
    
    # Merge with non-sku dataframe, keeping all rows from both dataframes
    master_skeleton = skuXdate_df.merge(nonsku_event_and_campaign_df, on = "daydate", how = "outer")
    master_skeleton['sku_id'] = master_skeleton['sku_id'].fillna('None') # On days where no SKU events took place sku_id is None
    master_skeleton = master_skeleton.fillna(0)  # All other NAN values are 0
    
    return master_skeleton


# ### 2.1.2 Shopify

# In[12]:


# this function adds a prefix to every non-airbyte_hashid column in the dataframe in order to track column source after merging
def rename_dfcolumns(df, prefix):
    for c in list(df.columns):
        if "airbyte" not in c:
            df.rename(columns={c:(prefix+c)}, inplace=True)


# #### 2.1.2.1 Shopify - Products

# In[13]:


# OLD,not used anymore: this function extracts all variant skus from a json column and converts all entries per row of the json parent row into a number of new single rows
def shopify_Extract_variant_sku_id(df, columnname):
    df_norm = df.copy()
    
    # normalize variants column
    df_norm = pd.concat([df_norm, pd.json_normalize(df_norm["variants"])], axis=1)

    # count number of variants per product family
    df_norm["number_of_variants"] = df_norm["variants"].apply(lambda x: len(x))

    # max number of skus
    max_variants = df_norm["number_of_variants"].max()
    
    # create list of columns for all columns containing names variant_1 .. to variant_N
    list_variant_columns = []
    
    # loop over number of items and extract variant skus in each column
    for i in range(max_variants):
        df_norm["variant_"+str(i+1)] = pd.json_normalize(df_norm[i]).sku
        list_variant_columns.append("variant_"+str(i+1))
        df_norm.drop(columns=[i], inplace=True)
    
    # get list of columns for id_vars for pd.melt by substracting list_variant_columns
    list_NOvariant_columns = [x for x in list(df_norm.columns) if x not in list_variant_columns]
    
    # melt down to get one row per variant
    df_norm = pd.melt(df_norm, id_vars = list_NOvariant_columns)
    
    # drop nan values after melt
    df_norm.dropna(subset=["value"], inplace = True)
    
    # drop nested column where information was extracted from
    df_norm.drop(columns=[columnname, "variable"], inplace=True)
            
    return df_norm


# #### 2.1.2.2 Shopify - Sales

# ## 2.2 Database connection and data import/export

# In[14]:


# A function that takes in a PostgreSQL query and outputs a pandas database 
def create_pandas_table(sql_query, database):
    table = pd.read_sql_query(sql_query, database)
    return table


# In[15]:


def import_data_from_PSQLDB(schema, table, engine, columns="*"):
    """ a sqlalchemy engine needs to be active for this function to work"""
    
    return pd.DataFrame(engine.execute("SELECT "+columns+" FROM "+schema+"."+table).fetchall(), columns=engine.execute("SELECT "+columns+" FROM "+schema+"."+table).keys())


# In[1]:


def import_data_AWSRDS(table, schema, engine):
    t = Timer(schema+table)
    chunks = list()
    for chunk in pd.read_sql("SELECT * FROM "+schema+"."+table, con=engine, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(list(chunks))
    t.end()
    return df

# Function that performs the unbundling by attributing metrics like past and future sales on the single items building the bundles
def unbundling(clientname, df, quantity_df):
    # Read in bundle matrix
    #matrix = pd.read_excel(r"Bundel-Product-Matrix-"+clientname+".xlsx", index_col="variant_sku")
    onedrive_api = OneDriveAPI(client_id, client_secret, tenant_id, msal_scope, site_id)
    matrix = onedrive_api.download_file_by_relative_path("NOGL_shared/"+clientname+"/Bundel-Product-Matrix-"+clientname+".xlsx")
    matrix.set_index("variant_sku", inplace=True)
    # Get names of single items
    items = matrix.columns
    # Get names of bundles
    bundles = matrix.index
    # Adding a boolean as flag if item is bundle or not
    df['isBundle'] = df['variant_sku'].isin(list(bundles))
    # Merge df with bundle matrix 
    df_bundles = df.merge(matrix,on="variant_sku")
    # Defining target columns
    target_columns = ['lineitems_quantity', 'revenue', 'VOIDS_forecast_q0', 'VOIDS_forecast_q1',
                    'VOIDS_forecast_q2', 'VOIDS_forecast_q3', 'VOIDS_forecast_q4',
                    'VOIDS_forecast_q5', 'VOIDS_forecast_q6']
    # Creating combinations of the items and the columns we are attributing to be able to assign it to the right item in the right column afterwards
    bundle_columns = [item + "~" + column + "~bundles" for item in items for column in target_columns]
    df_bundles_new = pd.DataFrame(columns=bundle_columns)

    # Calculating the metrices for each item and column using the bundle matrix
    for column in target_columns:
        for item in items:
            df_bundles_new[item + "~" + column + "~bundles"] = df_bundles[column] * df_bundles[item]
    # Merging the obtained df with the original one
    org_bundle_df = df_bundles[df_bundles["isBundle"]==True]
    df_bundles = org_bundle_df.merge(df_bundles_new, left_index=True, right_index=True)
    # Selecting the desired columns
    output_columns = df_bundles.iloc[:,-len(target_columns)*len(items):].columns
    # Creating a list of dates to loop over
    dates = df_bundles["daydate"].unique()
    # Creating a df with the single items
    df_single = df[df["isBundle"]==False]
    # Creating the target columns and setting them to 0
    df_single[target_columns] = 0
    # Attributing the metrics on the single items
    for date in dates:
        for column in output_columns:
            item = column.split("~")[0]
            columnstr = column.split("~")[1]
            mask = (df_single["variant_sku"]==item) & (df_single["daydate"]==date)
            df_single.loc[mask, columnstr] = df_bundles[column].loc[df_bundles["daydate"]==date].sum()
    # Renaming the columns to make them differentiable
    new_cols = {col: col + "_bundle" for col in target_columns}
    df_single = df_single.rename(columns=new_cols)
    # Loading the values for the single items and concat it to the df to have both single and bundle values for each item
    single_values = df[df['isBundle']==False][target_columns]
    df_single = pd.concat([df_single,single_values],axis=1)
    # Calculating a total value for each of the columns using the bundle and single values
    for column in target_columns:
        df_single[column+"_total"] = df_single.apply(lambda row: float(row[column] or 0.0) + float(row[column+"_bundle"] or 0.0), axis=1)
    # Creating the original bundle df and merging it with the single df to obtain the final df without the variant_inventory_quantity
    df_bundles = df.merge(matrix,on="variant_sku")
    final_df = pd.merge(df_bundles, df_single, on=['variant_sku'], how='outer')
    #engine = create_engine('postgresql://'+"postgres"+":"+"voids4thewin"+"@"+"voidsdb.c2wwnfcaisej.eu-central-1.rds.amazonaws.com"+":5432/"+"stoertebekker",echo=False)
    #quantity_df = import_data_AWSRDS(schema="transformed",table="shopify_products",engine=engine)
    # Merging the variant_inventory_quantity to the final df
    final_df = final_df.merge(quantity_df, on="variant_sku")
    
    # Renaming the columns
    for col in final_df.columns:
        if col.endswith('_x'):
            suffix_col = col.replace('_x', '_y')
            final_df[col.replace('_x', '')] = final_df[col].combine_first(final_df[suffix_col])
            final_df = final_df.drop(columns=[col, suffix_col])
        
    final_df = final_df.fillna(0)
    
    return final_df

# Function that calculates the date when a product is running out of stock
def DIO(df):
    # Creating columns for the new metrics
    # Column for the cumulative sum of the forecast
    df['cumsumforecast'] = 0
    # Column for the expected inventory, calculated by taking the difference
    # of the cumulative sum of the forecast and the "variant_inventory_quantity"
    df['inventoryPredicted'] = 0
    # Date when a item is running out of stock according to the calculation, if it does 
    df["outOfStockDate"] = pd.NaT
    
    # Looping through the item subsets of the dataframe to calculate: cumulative sum of the forecast,
    # expected inventory, date when a item is running out of stock
    for variant_sku in df['variant_sku'].unique():
        # Selecting subset of df
        df_sku = df[df['variant_sku'] == variant_sku]
        # Calculating the cumulative sum of the forecast for the 3rd quantile
        df_sku['cumsumforecast'] = df_sku['VOIDS_forecast_q3_total'].cumsum()
        # taking the difference of the cumulative sum of the forecast and the "variant_inventory_quantity"
        df_sku['inventoryPredicted'] = df_sku['variant_inventory_quantity'] - df_sku['cumsumforecast']
        # Finding the first date out of stock by searching for the first date with inventoryPredicted < 0
        df_sku["outOfStockDate"] = df_sku.loc[df_sku['inventoryPredicted']<0]["daydate"].min()
        # Update the original df with the changes
        df.update(df_sku)
   
    return df
        
        
# ## 2.2 Downloading from OneDrive

class OneDriveAPI:
    def __init__(self, client_id, client_secret, tenant_id, scope, site_id):
        """
        Initializes an instance of the OneDriveAPI class.

        Args:
            client_id (str): The client ID of the registered Azure AD application.
            client_secret (str): The client secret of the registered Azure AD application.
            tenant_id (str): The ID of the Azure AD tenant.
            scope (list[str]): The scopes required to access the OneDrive API.
            site_id (str): The ID of the SharePoint site containing the OneDrive folder.

        Returns:
            None

        Description:
            This function initializes an instance of the OneDriveAPI class. It sets the necessary instance variables, 
            and initializes the MSAL app instance for acquiring an access token.

        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scope = scope
        self.site_id = site_id
        self.access_token = None
        self.refresh_token = None
        self.token_expiration = None

        self.msal_app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )

        self._get_access_token()

    def _get_access_token(self):
        """
        Gets an access token for accessing the OneDrive API.

        Args:
            None

        Returns:
            None

        Description:
            This function gets an access token for accessing the OneDrive API. It first tries to acquire a token
            silently, and if that fails, it acquires a new token using the client credentials.

        """
        result = self.msal_app.acquire_token_silent(
            scopes=self.scope,
            account=None
        )

        if not result:
            result = self.msal_app.acquire_token_for_client(scopes=self.scope)

        if "access_token" in result:
            self.access_token = result["access_token"]
            self.refresh_token = result.get("refresh_token")
            self.token_expiration = time.time() + result.get("expires_in")
        else:
            raise Exception("No Access Token found")
        
    def get_access_token(self):
        """
        Gets an access token for accessing the OneDrive API.

        Args:
            None

        Returns:
            access_token (str): The access token for accessing the OneDrive API.

        Description:
            This function gets an access token for accessing the OneDrive API. It checks if the token has expired,
            and if it has, it acquires a new token using the refresh token.

        """

        if self.token_expiration is None or time.time() + 300 >= self.token_expiration:
            result = self.msal_app.acquire_token_by_refresh_token(
                refresh_token=self.refresh_token,
                scopes=self.scope,
            )

            if "access_token" in result:
                self.access_token = result["access_token"]
                self.refresh_token = result.get("refresh_token")
                self.token_expiration = time.time() + result.get("expires_in")
            else:
                raise Exception("Could not refresh Access Token")

        return self.access_token

    def find_item_by_relative_path(self, relative_path):
            """
            This function retrieves the item in OneDrive corresponding to the specified relative path.

            Arguments:
            relative_path -- The relative path to the item in OneDrive.

            Output:
            Returns a JSON object representing the retrieved item.
            """
            url = f"https://graph.microsoft.com/v1.0/drive/root:/{relative_path}"
            response = requests.get(url, headers={"Authorization": f"Bearer {self.get_access_token()}"})
            response.raise_for_status()
            return response.json()
    
    def download_file_by_relative_path(self, relative_path):
        """
        This function downloads the file specified in the relative path.

        Arguments:
        relative_path (str): The relative path of the file to download

        Output:
        Returns a pandas DataFrame containing the data from the downloaded file

        Description:
        This function downloads a file from OneDrive using its relative path and returns its contents as a pandas DataFrame.
        It first calls the find_item_by_relative_path method to get the ID of the file.
        It then uses the file ID to construct the URL of the file and sends an HTTP GET request to retrieve the file information.
        If the response status code is not 200, it raises an exception.
        It extracts the file name and download URL from the response data and sends an HTTP GET request to download the file.
        If the response status code is not 200, it raises an exception.
        It then uses pandas.read_excel or pandas.read_csv method to read the file data from the response content and returns the data as a pandas DataFrame.
        """
        item = self.find_item_by_relative_path(relative_path)
        file_id = item["id"]
        file_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{file_id}"
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        
        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to retrieve file information")
            
        file_data = response.json()
        file_name = file_data["name"]
        download_url = file_data["@microsoft.graph.downloadUrl"]
        
        response = requests.get(download_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to download file")
        
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
        elif file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(response.content), engine="python", index_col=0)
        else:
            raise Exception("File format not supported")
            
        return df
    
    def load_files_from_folder(self, relative_folder_path, filename_startswith):
        """
        This function loads all files from a OneDrive folder that match a specified prefix.

        Arguments:
        relative_folder_path (str): The relative path of the folder to load files from
        filename_startswith (str): The prefix of the filenames to match

        Output:
        Returns a pandas DataFrame containing the concatenated data from all matching files.

        Description:
        This function loads all files from a OneDrive folder that start with a specified prefix and returns their data concatenated as a single pandas DataFrame.
        It first calls the find_item_by_relative_path method to get the ID of the folder.
        It then uses the folder ID to construct the URL of the folder contents and sends an HTTP GET request to retrieve the contents.
        If the response status code is not 200, it raises an exception.
        It extracts the files from the response data and reads each file into a pandas DataFrame using pandas.read_csv or pandas.read_excel method depending on the file format.
        It then appends the data to a list of data frames and concatenates all data frames into a single data frame.
        """
        item = self.find_item_by_relative_path(relative_folder_path)
        folder_id = item["id"]
        folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{folder_id}/children"
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}

        response = requests.get(folder_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to retrieve folder contents")

        files = response.json()["value"]
        df_list = []
        for file in files:
            if file["name"].startswith(filename_startswith):
                download_url = file.get("@microsoft.graph.downloadUrl")
                if download_url is None:
                    continue

                response = requests.get(download_url, headers=headers)
                if response.status_code != 200:
                    raise Exception("Failed to download file")

                if file["name"].lower().endswith(".csv"):
                    df = pd.read_csv(BytesIO(response.content), engine="python", index_col=0)
                elif file["name"].lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(BytesIO(response.content), engine="openpyxl")
                else:
                    continue

                df_list.append(df)

                print(file["name"])

        df_concatenated = pd.concat(df_list, ignore_index=True)
        return df_concatenated
    

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


def load_main(name=None):
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

def load_pdp(name=None, engine=None):
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
    
    print("columns:", df.columns)
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
    Adds dates from 2020-01-01 until last date plus 200.
    Adds columns as np.nan if the date doesnt exist in df
    """
    
    df_dates = create_dateframe()

    # reverse to chrono order
    df_dates=df_dates.iloc[::-1]
    
    # get cols
    df_cols = list(df.columns)
    print(df_cols)
    data = {x:[] for x in df_cols if x != "ga_date"}
    data["date"] = []
    
    
    row_idx = 0
    df_rows = df.to_dict()
    len_df = df.shape[0]
  
    # for each date
    for date in df_dates["daydate"].astype(str):
        
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
    Adds dates from 2020-01-01 until last date plus 200
    for PDP, this function will create a date for each product


    Adds columns as np.nan if the date doesnt exist in df
    """
    
    df_dates = create_dateframe()
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
    for date in df_dates["daydate"].astype(str):
        
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

def merge_with_sceleton(df_main,df_pdp):
    """
    merge main and product into shopify sceleton
    returns a dataframe with the same number of rows than shopify sceleton
    """
    # load shopify
    df_sceleton = load_shopify_sceleton()
    #df_sceleton["daydate"]=df_sceleton["daydate"].apply(lambda x: x[0:11])
    df_sceleton["daydate"] = pd.to_datetime(df_sceleton["daydate"])
    df_sceleton["variant_sku"] = df_sceleton["variant_sku"].astype(str)
    
    # merge main 
    df_main.rename(columns={'date': 'daydate'}, inplace=True)
    df_main["daydate"] = pd.to_datetime(df_main["daydate"])
    df_main = df_main.rename({col:'main_'+ col for col in df_main.columns[~df_main.columns.isin(['daydate'])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_main,how='left', on='daydate')
    
    # merge pdp
    product_id = "variant_sku"
    df_pdp.rename(columns={'ga_date': 'daydate','ga_productsku':product_id}, inplace=True)
    df_pdp["daydate"] = pd.to_datetime(df_pdp["daydate"])
    df_pdp = df_pdp.rename({col:'pdp_'+ col for col in df_pdp.columns[~df_pdp.columns.isin(['daydate',product_id])]}, axis=1)
    df_sceleton=df_sceleton.merge(df_pdp,how='left', on=['daydate',product_id])
    
    df_sceleton.fillna(0, inplace=True)
    
    return df_sceleton



def check_dataset_columns(name,df1,df2):
    """
    Checks information of number of columns and columns order of both datasets
    
    Arguments:
        - name, a string to identify the dataset for messages only
        - df1, dataframe to test nr. 1
        - df2, dataframe to test nr. 2
    """
    print("\n"+"#"*80+"\n")
    
    # check number of columns
    if len(df1.columns) == len(df2.columns):
        print("["+name+"] SUCCESS - Both datasets have the same number of columns",len(df1.columns))
    else:
        print("["+name+"] FAIl -Datasets have a different number of columns")
        print("Main 1 has",len(df1.columns),"columns")
        print("Main 2 has",len(df2.columns),"columns")
    
    # check order
    if list(df1.columns.values.tolist()) == list(df2.columns.values.tolist()):
        print("["+name+"] SUCCESS - Same Order")
    else:
        print("["+name+"] FAIL - Different Order")
    
    # check for duplicate columns
    df1_cols = list(df1.columns)
    duplicate_col = False
    for i in range(len(df1_cols)-1):
        if df1_cols[i] in df1_cols[i+1:]:
            duplicate_col=True
            print("["+name+"] FAIL - Duplicate Column -",df1_cols[i])
    if not duplicate_col:
        print("["+name+"] SUCCESS - No duplicate columns")
    
    # show columns
    #print("\nDataset 1 columns",list(df1.columns.values.tolist()))
    #print("\nDataset 2 columns",list(df2.columns.values.tolist()))
    
    print("\n"+"#"*80+"\n")
    
    # test for main
    df1 = load_main("custom__report__airbyte__main_scd.csv")
    df1 = run_main_pipeline(df1)
    df2 = load_main("custom__report__airbyte__main_scd_other.csv")
    df2 = run_main_pipeline(df2)


    # check number of columns and order
    check_dataset_columns("MAIN",df1,df2)


    # test for pdp
    df1 = load_pdp("custom__report__airbyte_pdp_scd.csv")
    df1 = run_pdp_pipeline(df1)
    df2 = load_pdp("custom__report__airbyte_pdp_scd_other.csv")
    df2 = run_pdp_pipeline(df2)


    # check number of columns and order
    check_dataset_columns("PDP",df1,df2)
    
def test_sum_of_metrics(metric):
    # levels description
    level_2 = ['organicSearch','paidSearch','display', 'social', 'referral', 'direct', 'email', 'other']
    level_3 = {
        "social" : ["facebook","instagram","pinterest","tiktok","youtube","snapchat","rest"],
        "paidSearch": ["search","smartshopping","display","discovery","pmax","rest"],
        "organicSearch":  ["google","rest"],
        "display": ["google","rest"]
    }
    level_4 = ["paid","organic"]

    # confirm metric

    df1_original = load_main("custom__report__airbyte__main_scd.csv")
    df1 = run_main_pipeline(df1_original,remove_duplicates_activated=False)

    # original metric
    sum_users = df1_original["ga_"+metric].sum()

    # work to test
    sum_users_transformed = 0

    # check level 2
    total_2 = total_3 = total_4 = 0 
    for item_2 in level_2:
        value =  df1[item_2+"_"+metric].sum()
        total_2 += value

        if not item_2 in level_3:
            total_3 += value
            total_4 += value
            continue

        # check level 3
        for item_3 in level_3[item_2]:
            value = df1[item_2+"_"+item_3+"_"+metric].sum()
            total_3 += value

            if item_2 != "social":
                total_4 += value
                continue

            # check level 4
            for item_4 in level_4:
                value = df1[item_2+"_"+item_3+"_"+item_4+"_"+metric].sum()
                total_4 += value


    print("\n"+"#"*20, " Test ","#"*20)     
    message = "SUCCESS" if sum_users==total_2 else "FAIL"
    print("Level 2 check",metric,"-",message,sum_users,total_2)

    message = "SUCCESS" if sum_users==total_3 else "FAIL"
    print("Level 3 check",metric,"-",message,sum_users,total_3)

    message = "SUCCESS" if sum_users==total_4 else "FAIL"
    print("Level 4 check",metric,"-",message,sum_users,total_4)

    print("Duplicates removal was ignored for this test")
    
def test_null_values():
    # we have three identifier "date","productsku","productvariant"
    # check if data needs to be cleaned. NaN values are not acceptable. We need 0.00% NaN values per feature!
    # check for duplicates and remove / consolidate them
    df1 = load_main()
    df1 = run_main_pipeline(df1)
    df2 = load_pdp()
    df2 = run_pdp_pipeline(df2)

    print("\n"+"#"*20, " Test Main ","#"*20)   
    null_values_counter(df1)

    print("\n"+"#"*20, " Test PDP","#"*20)   
    null_values_counter(df2)


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
            
        STL_results = STL_results.append(plevel_results)

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




# selected only the columns that are needed for the amazon_sales transformation
# Creating the list of column names for each category

def get_columns_to_keep():
    """
    This function creates a list of columns that will be kept for the amazon_sales transformation.
    """

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
        #'lineitem_buyer_info_giftwrap_price_amount',
        #'lineitem_buyer_info_giftwrap_tax_amount'
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

    return columns_to_keep, hierarchy, dates, keep_as_is, feature_engineering, boolean #, one_hot_encoding