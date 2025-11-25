#!/usr/bin/env python
# coding: utf-8

# # 1. Imports & Options

# ## 1.1 External imports

# In[1]:


import json
import requests
import numpy as np
import pandas as pd
import os
from datetime import datetime, date, timedelta


# ## 1.2 Internal imports

# In[2]:




import sys
import os
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)

# import defined country codes

import defines
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
params = config(section=section)
from sqlalchemy import create_engine   



# ## 1.3 Options

# In[3]:


# pd.set_option('display.max_rows', None)# 


# # 2. Load Functions

# ## 2.1 Holiday Functions

# In[4]:


def download_public_holidays(years, countries = defines.country_list):
    '''
    Scrapes Nager.data API to download a list of public holidays
    Requires a list of years and a list of countries for which data should be downloaded
    Countries are defined by their two letter country code and by default are loaded from defines.py    
    '''
    
    holidays_df =  pd.DataFrame(columns=['date', 'localName', 'name', 'countryCode', 'fixed', 'global'])  # Initialise empty dataframe

    for country in defines.country_list:  # Loop over countries in country list
        for year in years:  # For each country, loop over years in year list 

            # Download data from API 
            url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
            response = requests.get(url)
            json_data = json.loads(response.text)
            api_holidays = pd.DataFrame.from_dict(json_data)
            api_holidays['date'] = pd.to_datetime(api_holidays['date'])
            
            api_holidays = api_holidays[['date', 'localName', 'name', 'countryCode', 'fixed', 'global']] # Keep only relevant columns
            
            holidays_df = pd.concat([holidays_df, api_holidays])  # Append new data at end of df 
            holidays_df.reset_index(drop=True, inplace=True)
    return holidays_df


# ## 2.2 Special Event (Marketing Day) Functions

# In[5]:


def preprocess_sales_events(important_sales_events = defines.important_sales_events, secondary_sales_events = defines.secondary_sales_events):
    '''
    Prepares data for important marketing days 
    Takes two lists of tuples, important_sales_events and secondary_sales_events, defaulting to the values given in defines.py 
    '''
    
    # Convert sales event data to dataframe and add bouleans
    imp_se = pd.DataFrame(important_sales_events, columns = ['Start' , 'End', 'Name'])  # Convert sales event data to dataframe
    imp_se['Important Sales Event'] = 1 
    imp_se['Secondary Sales Event'] = 0
    sec_se = pd.DataFrame(secondary_sales_events, columns = ['Start' , 'End', 'Name']) 
    sec_se['Important Sales Event'] = 0
    sec_se['Secondary Sales Event'] = 1
    
    # Combine Important and Secondary events
    sales_events = pd.concat([imp_se, sec_se])

    # Split Up dates so that each day is one row
    sales_events['Start'] = pd.to_datetime(sales_events['Start'], dayfirst=True)
    sales_events['End'] = pd.to_datetime(sales_events['End'], dayfirst=True)
    sales_events['Dates'] = [pd.date_range(x, y) for x , y in zip(sales_events['Start'],sales_events['End'])]    
    sales_events = sales_events.explode('Dates') 
    
    return sales_events


# ## 2.3 General Functions

# In[6]:


def create_dateframe(days_into_future=200, datacollection_startdate = '01-01-20'):
    '''
    Create empty date dataframe
    '''

    end = date.today() + timedelta(days=days_into_future)

    start_date, end_date = datacollection_startdate, date(int(end.strftime("%Y")), int(end.strftime("%m")) , int(end.strftime("%d")))

    date_range = pd.date_range(start_date, end_date, freq='d')
    dateframe = pd.DataFrame(date_range, columns= ["daydate"]).sort_values("daydate",ascending=False)
    dateframe.reset_index(drop=True, inplace=True)
    
    return dateframe

def create_holidays_and_marketing_day_feautures(
    years_for_holiday = [2019, 2020, 2021, 2022, 2023, 2024], 
    countries = defines.country_list, 
    important_sales_events = defines.important_sales_events, 
    secondary_sales_events = defines.secondary_sales_events):
    '''
    For each day creates a features indicating if the day is a public holiday, an important marketing day (e.g. Christmas) or a less important marketing day (e.g. Halloween)
    Takes a list of years for the public holidays, a country list (by default defined in defines.py) and two lists of tuples for the marketing days (also defined in defines.py)
    Returns a dataframe with one row for each day in the target period (defined by the create dataframe function) and 3 columns
    '''

    holidays = download_public_holidays(years_for_holiday)
    holidays['Holiday'] = 1
    holidays = holidays.groupby('date').first()
    holidays.reset_index(inplace = True)
    holidays = holidays[['date', 'Holiday']]
    holidays = holidays.rename(columns = {'Holiday':"external_holiday"})
    
    sales_events = preprocess_sales_events()
    for event in list(sales_events.Name.unique()):
        sales_events[event] = sales_events["Name"].apply(lambda x: event if x == event else "")
    sales_events.rename(columns={"Important Sales Event":"external_importantSalesEvent","Secondary Sales Event":"external_secondarySalesEvent"}, inplace=True)
    #sales_events = sales_events.groupby('Dates').first()
    sales_events.reset_index(inplace = True, drop=True)
    sales_events.drop(columns=['Start', 'End', 'Name'], inplace=True)
    print(sales_events.columns)
    
    timeXfeature_df = create_dateframe()
    timeXfeature_df = timeXfeature_df.merge(sales_events, left_on = 'daydate', right_on = 'Dates', how = 'left')
    timeXfeature_df["external_importantSalesEvent"] = timeXfeature_df["external_importantSalesEvent"].fillna(0)
    timeXfeature_df["external_secondarySalesEvent"] = timeXfeature_df["external_secondarySalesEvent"].fillna(0)
    timeXfeature_df = timeXfeature_df.fillna("")
    timeXfeature_df = timeXfeature_df.merge(holidays, left_on = 'daydate', right_on = 'date', how = 'left')
    timeXfeature_df.drop(columns=["date","Dates"], inplace=True)
    timeXfeature_df = timeXfeature_df.fillna(0)
    
    # eliminate duplicates coming from overlapping event data input
    for event in preprocess_sales_events().Name.unique():
        timeXfeature_df[event] = timeXfeature_df[['daydate',event]].groupby(['daydate'])[event].transform(lambda x: ''.join(x))
    for i in ["external_importantSalesEvent", "external_secondarySalesEvent", "external_holiday"]:
        timeXfeature_df[i] = timeXfeature_df[['daydate', i]].groupby(['daydate'])[i].transform(lambda x: 1 if x.sum() >=1 else 0)
    timeXfeature_df.drop_duplicates(inplace=True)
    
    return timeXfeature_df


# # 3 Preprocessing

# In[7]:


holidays_and_special_events_by_date = create_holidays_and_marketing_day_feautures()


# # 4 Read to database

# In[8]:


# from: https://stackoverflow.com/questions/58203973/pandas-unable-to-write-to-postgres-db-throws-keyerror-select-name-from-sqlit
# RDBMs including Postgres must use an SQLAlchemy connection for this method to create structures and append data. 
# However, read_sql does not require SQLAlchemy since it does not make persistent changes.

# create sqlalchemy connection

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"))


# In[9]:


# read dataframe to database schema transformed and replace entire table if existing already
holidays_and_special_events_by_date.to_sql("external_holidays_and_special_events_by_date", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")


# In[10]:


engine.dispose()


# In[11]:


print(params)

