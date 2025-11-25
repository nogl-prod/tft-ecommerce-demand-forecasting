#!/usr/bin/env python
# coding: utf-8

# # 1. Imports & Options

# ## 1.1 External imports

# In[1]:


import pandas as pd
import psycopg2
import os


# ## 1.2 Internal imports

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
from static_variables import *


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

# create sqlalchemy connection
from sqlalchemy import create_engine    

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)


# ## 1.3 Options

# In[4]:


pd.set_option('display.max_columns', None)


# # 2. Load Data

# In[5]:


# get productsxsceleton df from DB
table_name = "transformed"
productsxdays_sceleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)
productsxdays_sceleton = productsxdays_sceleton[['daydate', 'product_category', 'product_category_number', 'variant_sku', 'variant_id']]


# In[6]:
# Connect to one drive and save files into Plan Data directory 



# get plan data
client = args.client_name

onedrive_api = OneDriveAPI(client_id, client_secret, tenant_id, msal_scope, site_id)
relative_path = "NOGL_shared/"+client+"/"+client+"_Plan_Data_Weekly.xlsx"
weekly_plan_data = onedrive_api.download_file_by_relative_path(relative_path)
print(relative_path)
# Remove the old column index
weekly_plan_data.set_axis(weekly_plan_data.iloc[0], axis=1, inplace=True)
weekly_plan_data.drop(index=0, axis=0, inplace=True)

"""DATASOURCE_PLAN = "/opt/ml/processing/input/Data Source Transformations/Plan Data/"
filename = DATASOURCE_PLAN + client + "_Plan_Data_Weekly.xlsx"
print(filename)
# append this to the line below for production 
weekly_plan_data = pd.read_excel(filename, header = 1)  # Load Data"""


# In[7]:


engine.dispose()


# # 3. Define Functions

# ## 3.1 Preprocessing Functions

# In[8]:


def create_category_number_mapping(productsxdays_sceleton):
    category_mapping = productsxdays_sceleton.groupby(['product_category','product_category_number']).count().reset_index()
    category_mapping.drop(columns=["daydate", "variant_sku","variant_id"], inplace=True)
    return category_mapping


# In[9]:


def preprocess_plan_data(df, productsxdays_sceleton):
    '''
    Takes the weekly plan data provided by the customer and returns a dataframe containing daily budget allocations per product category
    This function is spepcific to the plan data provided by WeLive. It assumes that the first week only contains 3 days. Weekly budgets for all other 
    weeks is assumed to be on a 7 day basis.
    '''
    
    df=df.copy()
    
    # add product_category_number to weekly_plan_data
    df = df.merge(create_category_number_mapping(productsxdays_sceleton), how="left", on="product_category")

    # Create Isoweek Identifier 
    df['iso_week_id'] = df['First Day of the Week'].apply(lambda row: f'{row.isocalendar()[0]}-{row.isocalendar()[1]}') 
    
    # Loop over all budget columns and divide the weekly budget by the number of days in the week
    budgeting_columns = ["fb/insta_total_Budget", "googleads_total_Budget", "rest_total_budget", "All", "klaviyo_numberofcampaigns", "klaviyo_grossreach_perweek"] 
    for col in budgeting_columns:
        df['daily_'+col] = df[col] / 7  # Divide all weekly data by seven
        df.loc[df['iso_week_id'] == "2020-53", 'daily_'+col] = df[col] / 3  # Replace data for the first week by data divided by three
    
    # Rename columns with appropriate names for the model
    df.rename(columns = {"First Day of the Week" : "daydate", 
                         "daily_fb/insta_total_Budget" : "daily_FB_Budget", 
                         "daily_googleads_total_Budget" : "daily_GoogleLeads_Budget", 
                         "daily_rest_total_budget" : "daily_Other_Marketing_Budget", 
                         "daily_All" : "daily_Product_Sales_Target", 
                         "daily_klaviyo_numberofcampaigns" : "daily_num_planned_klaviyo_campaigns", 
                         "daily_klaviyo_grossreach_perweek" : "daily_planned_klaviyo_grossreach"}, inplace = True)
    
    # Drop all superfluous columns
    df = df[['daydate', 'iso_week_id', 'product_category_number', 'product_category', 'daily_FB_Budget', 'daily_GoogleLeads_Budget', 'daily_Other_Marketing_Budget', 'daily_Product_Sales_Target', 'daily_num_planned_klaviyo_campaigns', 'daily_planned_klaviyo_grossreach']]
    
    return df    


# ## 3.2 Data Merge Functions

# In[10]:


def merge_plan_data_to_product_x_day_skeleton(plan_data_df, skeleton):
    '''
    Takes the output of preprocess_plan_data and the product x day skeleton and returns a dataframe containing daily date for each sku for:
    The daily budget allocated to each product category for Facebook/Instagram, GoogleLeads, and other marketing channels 
    The daily sales target for each category 
    The number of klaviyo campaigns and expected reach for each category allocated on a daily basis
    All data is created by splitting weekly data into daily rates. 
    The number campaigns may be non-integer because the number of campaigns in divided by the number of days per week.
    '''
    
    # Create Isoweek Identifier 
    skeleton['iso_week_id'] = skeleton['daydate'].apply(lambda row: f'{row.isocalendar()[0]}-{row.isocalendar()[1]}') 
    
    # Merge daily plan data onto product x day skeleton via iso_week_id
    plan_data_df.drop(columns=["daydate", "product_category"], inplace=True)
    filled_skeleton = skeleton.merge(plan_data_df, on = ['iso_week_id', 'product_category_number'], how = 'left')
    
    # Fill all NAs with zero
    filled_skeleton = filled_skeleton.fillna(0)
    
    # Drop superfluous columns
    filled_skeleton = filled_skeleton[['daydate', 'variant_sku', 'variant_id', 'product_category_number',
                                       'daily_FB_Budget', 'daily_GoogleLeads_Budget','daily_Other_Marketing_Budget', 
                                       'daily_Product_Sales_Target', 'daily_num_planned_klaviyo_campaigns', 'daily_planned_klaviyo_grossreach']]
    
    return filled_skeleton


# # 4. Preprocessing

# In[11]:


daily_plan_data = preprocess_plan_data(weekly_plan_data, productsxdays_sceleton)
complete_plan_data = merge_plan_data_to_product_x_day_skeleton(daily_plan_data, productsxdays_sceleton)


# In[12]:


complete_plan_data.fillna(0, inplace=True)


# # 5. Export

# In[13]:


engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True


# In[14]:


t = Timer("Upload")
complete_plan_data.to_sql("marketingandsales_plan", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
t.end()


# In[15]:


engine.dispose()


# In[ ]:


print(params)

