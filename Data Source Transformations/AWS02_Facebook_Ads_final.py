#!/usr/bin/env python
# coding: utf-8

# # 1. Imports & Options

# ## 1.1 External imports 

# In[1]:


import numpy as np
import pandas as pd
from google.oauth2 import service_account
import pandas_gbq


# ## 1.2 Internal imports

import sys
import os
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)


# In[2]:

from configAWSRDS import config

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

# from support functions:

from Support_Functions import *


# In[3]:


pd.set_option('display.max_columns', None)


# ## 1.3 Support functions

# In[4]:


def download_query_from_bigquery(sql):
    '''
    Accesses Google BigQuery to download data depending on the passed query
    '''

    # Set Credential location
    credentials = service_account.Credentials.from_service_account_file(
        '/opt/ml/processing/input/01_Data Source Keys/Python/BigQuery Service Account Key/noglclientprojectwefriends-336852767362.json',
    )
    
    # Download Data
    df = pandas_gbq.read_gbq(sql, project_id="noglclientprojectwefriends", credentials=credentials)
    df.drop_duplicates(inplace=True)

    return df

def load_insights():
    
    # adjust section variable in FROM clause if specific schema should be loaded, now its loading dynamically based on client name
    df = download_query_from_bigquery("SELECT DISTINCT ad_id,clicks, date_start, frequency, impressions, inline_post_engagements , link_clicks, reach, unique_clicks, spend FROM `noglclientprojectwefriends." + section.replace("-","_") + "_facebookads.insights`")
    print("Downloaded BQ data with query: " + "SELECT DISTINCT ad_id,clicks, date_start, frequency, impressions, inline_post_engagements , link_clicks, reach, unique_clicks, spend FROM `noglclientprojectwefriends." + section.replace("-","_") + "_facebookads.insights`")
    
    # drop duplicates with same ad id on date -> Unique ad id and date are necessary
    # decision based on highest impressions (thats the value that is first in the funnel, so easiest to be triggered)
    df.sort_values(by=["date_start","ad_id","impressions"], ascending=False, inplace=True)
    df = df.groupby(["date_start","ad_id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("insights_df shape:", df.shape)

    return df

def load_ads():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.ads`")
    
    # drop duplicates with same ad id on date -> Unique ad id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["campaign_id","adset_id","id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["campaign_id","adset_id","id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("ads_df shape:", df.shape)
    return df

def load_adsets():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.ad_sets`")
    
    # drop duplicates with same ad id on date -> Unique adset id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["account_id", "campaign_id", "id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["account_id", "campaign_id", "id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("adsets_df shape:", df.shape)
    return df

def load_campaigns():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.campaigns`")
    
    # drop duplicates with same ad id on date -> Unique campaign id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["account_id", "id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["account_id", "id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("campaigns_df shape:", df.shape)
    return df

def preprocess_facebook_data(df):
    '''
    Takes a dataframe containing the insights datatable from the facebook ad data and returns a dataframe containingdaily data for:
    Totals and averages for: clicks, impressions, inline_post_engagement, link_clicks, reach, unique_clicks and spend; 
    The average frequency; and the number of active ads.
    '''
    
    '''
    Assumption: Every value missing in the data after the merge with the sceleton (NA values) are filled with 0. This is because a NA value will indicate
    that there was no ad active this day. This also applies to all mean/avg values.
    '''
    
    # Drop hours and minutes from date
    df['date_start'] = pd.to_datetime(df['date_start']).dt.date
    
    # Sum up data for each day
    sum_columns = ["clicks", "date_start", "impressions", "inline_post_engagements", "link_clicks", "reach", "spend", "unique_clicks"]  # Select columns to include in dataset
    fb_data_per_day = df[sum_columns].groupby("date_start").sum()  # Sum up results for each day
    fb_data_per_day.reset_index(inplace = True)  # Restore date column
    
    # Count number of Ads per day
    fb_ad_num = df[['date_start', 'ad_id']].groupby("date_start").count()  # Count ads each day
    fb_ad_num.reset_index(inplace=True)  # Restore date column
    fb_ad_num = fb_ad_num.rename(columns = {"ad_id" : "ads_per_day"})  # Set correct feature name
    
    # Add number of ads to dataframe
    fb_data_per_day = fb_data_per_day.merge(fb_ad_num, on = "date_start", how = "outer")    
    
    # Loop over columns in sum_columns ad create a new column with the average per ad 
    sum_columns.remove("date_start")  # remove non-numerical columns from list of features to average
    for column in sum_columns:  
        fb_data_per_day[column + "_avgPerAd"] = fb_data_per_day[column] / fb_data_per_day["ads_per_day"] # Divide Total per day by Number of Ads per day
    
    # Calculate average frequency
    fb_frequency_avg = df[["date_start", "frequency"]].groupby("date_start").mean()  # Average frequency on each day
    fb_frequency_avg.reset_index(inplace=True)  # Restore date column
    fb_frequency_avg = fb_frequency_avg.rename(columns = {"frequency" : "frequency_avg"})  # Rename feature column
    fb_data_per_day = fb_data_per_day.merge(fb_frequency_avg, on = "date_start", how = "outer")  # Add new data to final dataset
    
    return fb_data_per_day

def facebook_merge_into_product_x_day_skeleton(fb_data_df, skeleton_df):
    '''
    Merges the output of preprocess_facebook_data into the product x day skeleton
    '''
    
    fb_data_df = fb_data_df.rename(columns = {"date_start" : "daydate"})  # Rename date columns 
    skeleton_df = skeleton_df[['daydate', 
                               #"variant_sku", 
                               "variant_id"]]  # Drop superfluous columns
    skeleton_df['daydate'] = pd.to_datetime(skeleton_df['daydate']).dt.date
    skeleton_df = skeleton_df.merge(fb_data_df, how = "left", on = "daydate")   # Merge both datasets
    skeleton_df = skeleton_df.fillna(0)  # Fill all na's with zero
    
    return skeleton_df


# # 2. Load Data

# In[5]:


# get config params
  

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)


# In[6]:


productxday_skeleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)


# In[7]:


engine.dispose()


# In[8]:


insights_df = load_insights()
ads_df = load_ads()
adsets_df = load_adsets()
campaigns_df = load_campaigns()


# # 3. Create dataframe for Demand Shaping

# ## Rename columns

# In[9]:


# rename table features
df_dic = {"ads":ads_df, "adsets":adsets_df, "campaigns":campaigns_df}

for df in df_dic:
    for c in list(df_dic.get(df).columns):
        df_dic.get(df).rename(columns={c:df+"_"+c}, inplace=True)


# ## Get current info for ads, ad sets and campaigns

# In[10]:


campaigns_df_current = campaigns_df.sort_values(by=["campaigns_id", "campaigns_loaded_at", "campaigns_received_at"], ascending=[True, False, False])
campaigns_df_current = campaigns_df_current.groupby("campaigns_id", as_index=False).first()


# In[11]:


adsets_df_current = adsets_df.sort_values(by=["adsets_id", "adsets_loaded_at", "adsets_received_at"], ascending=[True, False, False])
adsets_df_current = adsets_df_current.groupby("adsets_id", as_index=False).first()


# In[12]:


ads_df_current = ads_df.sort_values(by=["ads_id", "ads_loaded_at", "ads_received_at"], ascending=[True, False, False])
ads_df_current = ads_df_current.groupby("ads_id", as_index=False).first()


# In[13]:


# generate id_mapping dictionary for ads, ad sets and campaigns
id_mapping_current = adsets_df_current.merge(ads_df_current, how="left", left_on="adsets_id", right_on="ads_adset_id").drop_duplicates()
id_mapping_current = id_mapping_current.merge(campaigns_df_current, how="left", left_on="adsets_campaign_id", right_on="campaigns_id").drop_duplicates()
id_mapping_current = id_mapping_current[["campaigns_account_id",
                                         "campaigns_id",
                                         "campaigns_name",
                                         "campaigns_effective_status",
                                         "adsets_id",
                                         "adsets_name",
                                         "adsets_effective_status",
                                         "adsets_daily_budget",
                                         "ads_id",
                                         "ads_name",
                                         "ads_status",
                                         "ads_bid_amount",
                                         "ads_bid_type"
                                         ]]


# In[14]:


# build a mapping list to find in the campaign / adset / ad names

unique_product_categories = list(productxday_skeleton.product_category.unique())

# add special names
to_append = ["Mug","Shirt"]
for a in to_append:
    unique_product_categories.append(a)
    
unique_product_categories.append("Rest")


# In[15]:


id_mapping_current["extract_campaign"] = id_mapping_current.campaigns_name.astype(str)
id_mapping_current["extract_adset"] = id_mapping_current.adsets_name.astype(str)
id_mapping_current["extract_ad"] = id_mapping_current.ads_name.astype(str)
id_mapping_current.head(3)


# In[16]:


# check if the strings unique_product_categories from can be found in the campaign / adset / ad names
for p in unique_product_categories:
    id_mapping_current["extract_campaign"] = id_mapping_current["extract_campaign"].apply(lambda x: str(p) if (str(p) in x) else x)
    id_mapping_current["extract_adset"] = id_mapping_current["extract_adset"].apply(lambda x: str(p) if (str(p) in x) else x)
    id_mapping_current["extract_ad"] = id_mapping_current["extract_ad"].apply(lambda x: str(p) if (str(p) in x) else x)
    
id_mapping_current["extract_campaign"] = id_mapping_current["extract_campaign"].apply(lambda x: x if (x in unique_product_categories) else "")
id_mapping_current["extract_adset"] = id_mapping_current["extract_adset"].apply(lambda x: x if (x in unique_product_categories) else "")
id_mapping_current["extract_ad"] = id_mapping_current["extract_ad"].apply(lambda x: x if (x in unique_product_categories) else "")


# In[17]:


# merge campaign / adset / ad names findings into one column

def pick_extract(extract_campaign, extract_adset, extract_ad):
    if extract_campaign != "":
        return extract_campaign
    elif extract_adset != "":
        return extract_adset
    elif extract_ad != "":
        return extract_ad
    else:
        return "Rest"
    
id_mapping_current["mapping"] = id_mapping_current.apply(lambda x: pick_extract(x.extract_campaign, x.extract_adset, x.extract_ad), axis=1)

id_mapping_current.drop(columns=["extract_campaign", "extract_adset", "extract_ad"], inplace=True)

# fill in missing values in mapping with "Rest"


# In[18]:


# change back special names
id_mapping_current["mapping"] = id_mapping_current["mapping"].str.replace('Shirt','T-Shirt')
id_mapping_current["mapping"] = id_mapping_current["mapping"].str.replace('Mug','Tasse')


# In[19]:


id_mapping_current


# In[20]:


# merge id information on insights_df
spends = insights_df.merge(id_mapping_current, how="left", left_on="ad_id", right_on="ads_id")


# # 4. Preprocess and create dataframe for Demand Forecasting

# In[21]:


fb_daily_data = preprocess_facebook_data(insights_df)
complete_facebook_data = facebook_merge_into_product_x_day_skeleton(fb_daily_data, productxday_skeleton)


# # 5. Export

# In[22]:


engine.dispose()


# In[23]:


engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True


# In[24]:


t = Timer("Upload")
complete_facebook_data.to_sql("facebook_ads", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
spends.to_sql("facebook_ad_spends", con = engine, schema="demand_shaping", if_exists='replace', index=False, chunksize=1000, method="multi")
t.end()


# In[25]:


engine.dispose()


# In[26]:


print(params)

