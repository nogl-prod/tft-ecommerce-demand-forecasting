#!/usr/bin/env python
# coding: utf-8

# # 1. Imports and Options

# ## 1.1 External imports

# In[1]:

#pd.DataFrame([["hallo",len(complete_klaviyo_data)]], columns=["step","number of rows"])

import pandas as pd
from datetime import datetime, date, timedelta
import json
import time


# ## 1.2 Internal Imports

# In[2]:

import sys
import os
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)

# import the 'config' funtion from the config.py file

from configAWSRDS import config

# from support functions:

from Support_Functions import *


# ## 1.3 Options

# In[3]:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)

# # 2. Load Data

# In[4]:


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


# In[5]:


# create sqlalchemy connection
from sqlalchemy import create_engine    

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)

print("engine is created:", engine)


# In[6]:


# Load Klaviyo Data
klaviyo_events_raw = import_data_AWSRDS(schema="klaviyo",table="events_scd",engine=engine)

klaviyo_campaigns_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns",engine=engine)

klaviyo_campaigns_lists_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns_lists",engine=engine)

klaviyo_lists_raw = import_data_AWSRDS(schema="klaviyo",table="lists",engine=engine)


# Load Product ID Data
klaviyo_productsxdays_sceleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)



# # Load Historical Data
# klaviyo_historical_data = import_data_AWSRDS(schema="historics",table="klaviyo_events_history",engine=engine)

# In[7]:


# change dtype of event_properties
klaviyo_events_raw["event_properties"]= klaviyo_events_raw["event_properties"].astype("string")

# In[8]:


#engine.dispose()


# # 3. Run pipeline

# In[9]:


def klaviyo_create_productxday_skeleton (events_raw_df, campaigns_raw_df, lists_df, campaigns_list_df, product_X_day_skeleton):
    '''
    Requires 5 input data frames (events_raw_df, campaigns_raw_df, lists_df, campaigns_list_df and product_X_day_skeleton) and 
    returns 1 dataframe  containing day-by-day data for each product. 
    '''
    
    # select only the correct columns from the event df
    klaviyo_events_raw = events_raw_df[['id',
                                             'uuid',
                                             'flow_id',
                                             'campaign_id',
                                             'statistic_id',
                                             'flow_message_id',
                                             'datetime',
                                             'timestamp',
                                             'event_name',
                                             'event_properties']]
    
    # Split events by type 
    # SKU events need to be handled seperately because events_properties json is not always build the same
    klaviyo_events_dict = split_event_data(klaviyo_events_raw)

    # Create SKU event dataframes
    klaviyo_events_sku_checkoutStarted = klaviyo_events_dict['checkout_started']
    klaviyo_events_sku_orderedProduct = klaviyo_events_dict['ordered_product']
    klaviyo_events_sku_cancelledOrder = klaviyo_events_dict['cancelled_order']
    klaviyo_events_sku_refundedOrder = klaviyo_events_dict['refunded_order']

    # Create non SKU event dataframe
    klaviyo_events_nonsku = klaviyo_events_dict['non_sku_events']
    
    # Preprocess SKU events
    checkoutStarted_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_checkoutStarted, "Checkout Started")
    cancelledOrder_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_cancelledOrder, "Cancelled Order")
    refundedOrder_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_refundedOrder, "Refunded Order")
    orderedProduct_skus_by_date = klaviyo_Extract_sku_id_from_orderedProduct(klaviyo_events_sku_orderedProduct, "Ordered Product")
    
    # Preprocess capaign data and non-sku events
    nonsku_event_and_campaign_by_date = preprocess_campaigns_and_nonsku_events(campaigns_raw_df, lists_df, campaigns_list_df, klaviyo_events_nonsku)
    
    # Prepare dataframes for merging
    klaviyo_productsxdays_sceleton = product_X_day_skeleton[['variant_sku', 'daydate', 'variant_id']]  # Drop all non-index columns 
    klaviyo_productsxdays_sceleton['daydate'] = pd.to_datetime(klaviyo_productsxdays_sceleton['daydate']) # Convert to datetime    
    
    
    klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.merge(nonsku_event_and_campaign_by_date, on = "daydate", how = "left")
    
    # Drop superfluous event_name columns
    checkoutStarted_skus_by_date = checkoutStarted_skus_by_date.drop("event_name", axis = 1)
    cancelledOrder_skus_by_date = cancelledOrder_skus_by_date.drop("event_name", axis = 1)
    refundedOrder_skus_by_date = refundedOrder_skus_by_date.drop("event_name", axis = 1)
    orderedProduct_skus_by_date = orderedProduct_skus_by_date.drop("event_name", axis = 1)
    
    skuXdate_df = checkoutStarted_skus_by_date.merge(cancelledOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    skuXdate_df = skuXdate_df.merge(refundedOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    skuXdate_df = skuXdate_df.merge(orderedProduct_skus_by_date, on =['sku_id', 'datetime'], how='outer')
    
    # prepare columns for merging
    skuXdate_df.rename(columns = {'datetime' : 'daydate', 'sku_id' : 'variant_sku'}, inplace = True) 
    skuXdate_df['daydate'] = pd.to_datetime(skuXdate_df['daydate'])
    
    # Clean event data
    skuXdate_df = skuXdate_df.fillna(0)
    skuXdate_df = skuXdate_df.drop_duplicates()

    # Fill product x day Skeleton with klaviyo data
    klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.merge(skuXdate_df, on = ['daydate', 'variant_sku'], how = 'left')
    klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.fillna(0) # fill all created NANs with zero

    return klaviyo_productsxdays_sceleton


# # 3. Preprocessing 

# ## 3.1 Airbyte Data Preprocessing

# In[10]:


# Create the current klaviyo from the airbyte datasets 
filled_productsxdays_sceleton = klaviyo_create_productxday_skeleton(klaviyo_events_raw, klaviyo_campaigns_raw, klaviyo_lists_raw, klaviyo_campaigns_lists_raw, klaviyo_productsxdays_sceleton)
# ## 3.2 Historical Data Integration

# klaviyo_historical_data['daydate'] = pd.to_datetime(klaviyo_historical_data['daydate'])  # Correct dtype for date
# 
# # Combine historical and airbyte data
# historical_segment = klaviyo_historical_data[klaviyo_historical_data['daydate'] <= '2022-06-02']  # All date prior to the 3rd of June is taken from the historical data
# airbyte_segment = filled_productsxdays_sceleton[filled_productsxdays_sceleton['daydate'] > '2022-06-02'] # All data past the 3rd of June is taken from the new airpyte data
# complete_klaviyo_data = pd.concat([airbyte_segment, historical_segment], axis = 0)  # Merge both datasets row wise to create one single data set

# In[11]:


complete_klaviyo_data = filled_productsxdays_sceleton.copy()


# ## 3.3 Rename features

# In[12]:


complete_klaviyo_data.rename(columns={'num_recipients':'num_recipients',
                                         'campaigns_sent':'campaigns_sent',
                                         'planned_recipients':'planned_recipients',
                                         'Campaign - Bounced Email':'campaign_BouncedEmail',
                                         'Campaign - Clicked Email':'campaign_ClickedEmail',
                                         'Campaign - Dropped Email':'campaign_DroppedEmail',
                                         'Campaign - Marked Email as Spam':'campaign_MarkedEmailasSpam',
                                         'Campaign - Opened Email':'campaign_OpenedEmail',
                                         'Campaign - Received Email':'campaign_ReceivedEmail',
                                         'Campaign - Unsubscribed':'campaign_Unsubscribed',
                                         'Flow - Bounced Email':'flow_BouncedEmail',
                                         'Flow - Clicked Email':'flow_ClickedEmail',
                                         'Flow - Dropped Email':'flow_DroppedEmail',
                                         'Flow - Marked Email as Spam':'flow_MarkedEmailasSpam',
                                         'Flow - Opened Email':'flow_OpenedEmail',
                                         'Flow - Received Email':'flow_ReceivedEmail',
                                         'Flow - Unsubscribed':'flow_Unsubscribed',
                                         'Other - Active on Site':'other_ActiveonSite',
                                         'Other - Fulfilled Order':'other_FulfilledOrder',
                                         'Other - Fulfilled Partial Order':'other_FulfilledPartialOrder',
                                         'Other - Placed Order':'other_PlacedOrder',
                                         'Other - Subscribed to List':'other_SubscribedtoList',
                                         'Other - Unsubscribed from List':'other_UnsubscribedfromList',
                                         'Other - Updated Email Preferences':'other_UpdatedEmailPreferences',
                                         'Campaign - Active on Site':'campaign_ActiveonSite',
                                         'Campaign - Placed Order':'campaign_PlacedOrder',
                                         'Campaign - Fulfilled Order':'campaign_FulfilledOrder',
                                         'Campaign - Subscribed to List':'campaign_SubscribedtoList',
                                         'Campaign - Fulfilled Partial Order':'campaign_FulfilledPartialOrder',
                                         'Campaign - Unsubscribed from List':'campaign_UnsubscribedfromList',
                                         'Campaign - Updated Email Preferences':'campaign_UpdatedEmailPreferences',
                                         'Flow - Active on Site':'flow_ActiveonSite',
                                         'Flow - Placed Order':'flow_PlacedOrder',
                                         'Flow - Fulfilled Order':'flow_FulfilledOrder',
                                         'Flow - Subscribed to List':'flow_SubscribedtoList',
                                         'Flow - Fulfilled Partial Order':'flow_FulfilledPartialOrder',
                                         'Flow - Unsubscribed from List':'flow_UnsubscribedfromList',
                                         'Flow - Updated Email Preferences':'flow_UpdatedEmailPreferences',
                                         'Other - Received Email':'other_ReceivedEmail',
                                         'Other - Opened Email':'other_OpenedEmail',
                                         'Other - Unsubscribed':'other_Unsubscribed',
                                         'Other - Clicked Email':'other_ClickedEmail',
                                         'Other - Bounced Email':'other_BouncedEmail',
                                         'Other - Marked Email as Spam':'other_MarkedEmailasSpam',
                                         'Other - Dropped Email':'other_DroppedEmail',
                                         'Checkout Started':'CheckoutStarted',
                                         'Cancelled Order':'CancelledOrder',
                                         'Refunded Order':'RefundedOrder',
                                         'Ordered Product':'OrderedProduct'}, inplace=True)


# # 4. Export

# In[15]:


t = Timer("Export")


# In[16]:


complete_klaviyo_data.head(5).to_sql('klaviyo', con = engine, schema="transformed", if_exists='replace',index=False) #drops old table and creates new empty table


# In[17]:


engine.execute('TRUNCATE TABLE transformed.klaviyo') #Truncate the table in case you've already run the script before


# In[18]:


engine.dispose()


# In[19]:


complete_klaviyo_data.to_csv('/opt/ml/processing/input/Data Source Transformations/Data Uploads/klaviyo.csv', index=False, header=False) #Name the .csv file reference in line 29 here

# In[25]:


col_names = ", ".join(map(lambda x: '"'+x+'"', list(complete_klaviyo_data.columns)))


# In[26]:


with open('/opt/ml/processing/input/Data Source Transformations/Data Uploads/klaviyo.csv', 'r') as f:    
    conn = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database")).raw_connection()
    cursor = conn.cursor()
    cmd = 'COPY transformed.klaviyo('+ col_names+ ')'+ 'FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
    cursor.copy_expert(cmd, f)
    conn.commit()



# In[27]:


t.end()


# In[28]:


engine.dispose()


# In[29]:


print(params)

