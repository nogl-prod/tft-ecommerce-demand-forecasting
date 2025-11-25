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
from geopy.geocoders import Nominatim
import time
import psycopg2


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


# from support functions:

from Support_Functions import create_pandas_table, rename_dfcolumns, create_dateframe
from static_variables import datacollection_startdate


# ## 1.3 Options

# In[3]:


#pd.set_option('display.max_columns', None)


# # 2. Load Fundtions

# ## 2.1 Weather Forecast Functions

# def download_weather_forecast(API_KEY, city_list = defines.city_list, metric=True):
#     '''
#     Downloads 8 day weather forecast for cities in city_list
#     Requires OpenWeatherMap API_KEY and returns a dataframe containing humidity, rain, max temperature and minimum temperature data for each city
#     By default, temperatures are in Kelvin, use metric=True to download Celsius data
#     By default uses cities from city_list defined in defines.py
#     '''
#     
#     # Initialise locator to get coordinates from city names
#     geolocator = Nominatim(user_agent="MyApp")
# 
#     # Create an empty dateframe to merge on
#     start_date = datetime.now()  # Weather forecast starts today
#     end_date = start_date + timedelta(days=7) # Weather forecast ends 7 days from now (on day 8)
#     
#     date_range = pd.date_range(start_date, end_date, freq='d')  # Create the dates between start and end date
#     weather_merged = pd.DataFrame(date_range, columns= ["daydate"]).sort_values("daydate",ascending=False) # create dataframe from date range 
#     weather_merged = weather_merged['daydate'].dt.date  # drop hours and minutes
#     weather_merged = pd.to_datetime(weather_merged).to_frame() # convert to datetime
#     weather_merged.reset_index(drop=True, inplace=True)  # reset indexes
# 
#     api_weather = {} # Initialise an empty dictionary to fill with data for each city
#     
#     # Loop over cities to download their weather data
#     for city in city_list: 
#         location = geolocator.geocode(city)  # Get coordinates for city
#         lat = location.latitude  # Save longitude
#         lon = location.longitude  # Save Latitude 
# 
#         
#         url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid=' + API_KEY
#         if metric == True:  # If metric is set to true, download data in Celsius instead of Kelvin
#             url = url + '&units=metric'
# 
#         # Access the API
#         response = requests.get(url)
#         json_data = json.loads(response.text)
#         api_weather[city] = pd.json_normalize(json_data['daily'])
#         
#         # Preprocess Data
#         api_weather[city]['dt'] = api_weather[city]['dt'].apply(datetime.fromtimestamp).dt.date  # Convert UNIX time to readable format
#         api_weather[city]['dt'] = pd.to_datetime(api_weather[city]['dt'])
#         
#         api_weather[city] = api_weather[city][['dt', 'humidity', 'rain', 'temp.min', 'temp.max']] # Keep only relevant columns
#         
#         # In case of missing data, fill with mean (humidity and temperature) or zero (rain)
#         # Note: Rain is always NAN if there is no rainfall expected
#         api_weather[city]['humidity'] = api_weather[city]['humidity'].fillna(api_weather[city]['humidity'].mean())
#         api_weather[city]['rain'] = api_weather[city]['rain'].fillna(0)
#         api_weather[city]['temp.min'] = api_weather[city]['temp.min'].fillna(api_weather[city]['temp.min'].mean())
#         api_weather[city]['temp.max'] = api_weather[city]['temp.max'].fillna(api_weather[city]['temp.max'].mean())
# 
#         # Rename columns to include city name
#         api_weather[city].rename(columns={'dt': 'daydate', 
#                                                    'humidity': f'{city}_humidity', 
#                                                    'rain': f'{city}_rain', 
#                                                    'temp.min':f'{city}_temp.min', 
#                                                    'temp.max':f'{city}_temp.max'}, inplace=True)
# 
#         # Merge city data into master dataframe
#         weather_merged = weather_merged.merge(api_weather[city], on='daydate', how='left')
# 
#     return weather_merged

# ## 2.2 Visualcrossing Weather Functions

# In[4]:


def download_weather_data(API_KEY, city_list = defines.city_list, start_date = '2020-01-01', end_date = datetime.now(), safety_check = False):
    '''
    This function takes one city and returns a dataframe containing the historical, forecast and statistical forecast (beyond 15 days) temperature, humidity and precipitation data for all cities set in city_list 
    WAARNING: This function makes ~980 request to the visualcrossing weather API. visualcrossing offers 1000 free records per day. 
    Running this function more than once per day will incur a cost!
    To avoid accidents the function is disabled by default and needs to be enabled by setting safety_check = True
    Requires an API key and a city (string), start_date and end_date are optional parameters. 
    '''
    if type(end_date) != str:
        end_date = end_date + timedelta(days=200)
        end_date = date(int(end_date.strftime("%Y")), int(end_date.strftime("%m")) , int(end_date.strftime("%d")))
    
    date_range = pd.date_range(start_date, end_date, freq='d')  # Create the dates between start and end date
    weather_merged = pd.DataFrame(date_range, columns= ["daydate"]).sort_values("daydate",ascending=False) # create dataframe from date range 
    weather_merged = weather_merged['daydate'].dt.date  # drop hours and minutes
    weather_merged = pd.to_datetime(weather_merged).to_frame() # convert to datetime
    weather_merged.reset_index(drop=True, inplace=True)  # reset indexes

    api_weather = {} # Initialise an empty dictionary to fill with data for each city
    
    for city in city_list:
    
        if safety_check == False:  # safety check
            return print("Running this function more than once a day will incur a financial cost. If you're sure you want to run the function, set safety_check = True")

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=json"

        # Querry API
        response = requests.get(url)
        print(response)
        json_data = json.loads(response.text)

        api_weather[city] = pd.json_normalize(json_data['days']) # Access daily date
        api_weather[city] = api_weather[city][['datetime', 'humidity', 'precip', 'tempmin', 'tempmax']]   # Drop superfluous columns
        api_weather[city]['datetime'] = pd.to_datetime(api_weather[city]['datetime'])

        # Rename columns to inlcude city name 
        api_weather[city].rename(columns={'datetime': 'daydate', 
                                                   'humidity': f'{city}_humidity', 
                                                   'precip': f'{city}_rain', 
                                                   'tempmin':f'{city}_temp.min', 
                                                   'tempmax':f'{city}_temp.max'}, inplace=True)
        
        # Merge city data into master dataframe
        weather_merged = weather_merged.merge(api_weather[city], on='daydate', how='left')
        
    return weather_merged


# # 3. DB connection settings

# In[5]:


# get config params
#params = config()

# create sqlalchemy connection
#from sqlalchemy import create_engine    

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)


# #### Run this once to initiate historic download start date

# """ create latest sync date (ONE TIME ACTIVITY) """
# start = pd.to_datetime("2020-01-01")
# start = date(int(start.strftime("%Y")), int(start.strftime("%m")) , int(start.strftime("%d")))
# pd.DataFrame([[start,True]], columns=["latest_sync_date","first_time_sync"]).to_sql("latest_sync", con = engine, schema="external_weather", if_exists='replace', index=False)

# # 4. Run pipeline (incl. saving results)

# In[6]:


# Set visualcrossing API key from environment variable
visualcrossing_apikey = os.getenv("VISUALCROSSING_API_KEY", "")
if not visualcrossing_apikey:
    raise ValueError(
        "Missing VISUALCROSSING_API_KEY environment variable. "
        "Please set it in your .env file or environment."
    )


# In[7]:


# get latest date to download data for
historic_weather_table_name = "external_weather"
latest_sync = pd.DataFrame(engine.execute("SELECT * FROM "+historic_weather_table_name+".latest_sync").fetchall(), columns=engine.execute("SELECT * FROM "+historic_weather_table_name+".latest_sync").keys())
latest_date = latest_sync.iloc[0,0]


# In[8]:


# add one day to latest_date if not first time sync
if latest_sync.first_time_sync.iloc[0] == False:
    latest_date = (pd.to_datetime(latest_date) + timedelta(days=1))
    latest_date = date(int(latest_date.strftime("%Y")), int(latest_date.strftime("%m")) , int(latest_date.strftime("%d")))


# In[9]:


# download new data
weather_data_new = download_weather_data(API_KEY=visualcrossing_apikey, safety_check = True, start_date = latest_date)


# In[10]:


# for all non first times, get history and combine with updates, then update latest_sync
if latest_sync.first_time_sync.iloc[0] == False:
    historic_weather = pd.DataFrame(engine.execute("SELECT * FROM "+historic_weather_table_name+".weather").fetchall(), 
                                columns=engine.execute("SELECT * FROM "+historic_weather_table_name+".weather").keys())
    
    # combine history with newest data
        # cut history from last sync date
    historic_weather = historic_weather[pd.to_datetime(historic_weather["daydate"]) < pd.to_datetime(latest_date)]
        # append two dataframes
    weather = weather_data_new.append(historic_weather)
    
    # save results to db
    weather.to_sql("weather", con = engine, schema="external_weather", if_exists='replace', index=False, chunksize=1000, method="multi")
    weather.to_sql("external_weather", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
    
    # update latest_sync
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = date(int(yesterday.strftime("%Y")), int(yesterday.strftime("%m")) , int(yesterday.strftime("%d")))
    latest_sync = pd.DataFrame([[yesterday,False]], 
                               columns=["latest_sync_date","first_time_sync"])
    latest_sync.to_sql("latest_sync", con = engine, schema="external_weather", if_exists='replace', index=False, chunksize=1000, method="multi")
    
    
# for first time sync put in history = new data and update latest_sync to todays date
if latest_sync.first_time_sync.iloc[0] == True:
    weather = weather_data_new.copy()

    # save results to db
    weather.to_sql("weather", con = engine, schema="external_weather", if_exists='replace', index=False, chunksize=1000, method="multi")
    weather.to_sql("external_weather", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")


    # update latest_sync
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = date(int(yesterday.strftime("%Y")), int(yesterday.strftime("%m")) , int(yesterday.strftime("%d")))
    latest_sync = pd.DataFrame([[yesterday,False]], 
                               columns=["latest_sync_date","first_time_sync"])
    latest_sync.to_sql("latest_sync", con = engine, schema="external_weather", if_exists='replace', index=False, chunksize=1000, method="multi")


# In[11]:


engine.dispose()


# In[12]:


print(params)

