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

from static_variables import *


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


# create empty date dataframe
def create_dateframe(datacollection_startdate = "2020-01-01", days_into_future=200):
    """
    Creates a dataframe with one column "date" starting on datacollection_startdate 
    until today plus days_into_future
    """
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


def preprocess_campaigns_and_nonsku_events (campaigns_df, lists_df, campaigns_lists_df, nonsku_df, nonsku_preprocessed = False):
    """
    Takes the campaigns, lists, campaigns_lists and non-sku event data and returns a day-by-day dataframe from datacollection_startdate to today + days_into_future 
    Output inlcudes data on number of recipients, number of campaigns and number of planned recipients (people in lists) as well as data on non-sku events
    """
    master_df = create_dateframe()  # Create empty dataframe

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
    target_columns = ['lineitems_quantity', 'revenue', 'NOGL_forecast_q0', 'NOGL_forecast_q1',
                    'NOGL_forecast_q2', 'NOGL_forecast_q3', 'NOGL_forecast_q4',
                    'NOGL_forecast_q5', 'NOGL_forecast_q6']
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
        df_sku['cumsumforecast'] = df_sku['NOGL_forecast_q3_total'].cumsum()
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