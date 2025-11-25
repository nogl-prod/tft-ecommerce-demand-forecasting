#!/usr/bin/env python
# coding: utf-8

# # 1. Imports & Options

# ## 1.1 External imports

# In[1]:


import pandas as pd
import pycountry
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




pd.set_option('display.max_columns', None)


# # 2. Load Fundtions

# ## 2.1 COVID Case Functions

# In[4]:


def preprocess_covid_data(country_list = defines.country_list):
    '''
    Takes a list of two letter country codes and returns a dataframe containing daily covid data for each country 
    By default the list of countries is country_list from defines.py
    Covid data includes 7 day rolling averages for new cases and new death, as well as the percentage of the population that is vaccinated
    If any previous data exists, missing data is filled in with the most recent value. If there is no previous data missing values are assumed to be zero.
    Values past the end of the source data set remain as NaN (intended to be filled in when the data is merged)
    '''
    
    # Download data from datasource
    raw_covid_data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
    
    # Convert country abbreviations into full country names (DE -> Germany)
    full_name_countries = []
    for country in country_list:
        full_name_countries.append(pycountry.countries.get(alpha_2=country).name)
    
    # Keep only relevant data
    covid_data = raw_covid_data[raw_covid_data['location'].isin(full_name_countries)]  # drop all countries not in country_list
    covid_data = covid_data[['location', 'date', 'new_cases_smoothed_per_million',  # keep only chosen covid data columns
                                                    'new_deaths_smoothed_per_million',
                                                    'people_vaccinated_per_hundred',
                                                    'weekly_icu_admissions_per_million',
                                                    'new_tests_smoothed_per_thousand',
                                                    'median_age',
                                                    'gdp_per_capita',
                                                    'human_development_index',
                                                    'hosp_patients_per_million',
                                                    'hospital_beds_per_thousand']]


    covid_data['date'] = pd.to_datetime(covid_data['date'])
    
    date_df = create_dateframe() # create day-by-day dataframe
    
    # Engineer feature for capacity of hospital beds per million
    covid_data['hosp_beds_capacity_per_million'] = (covid_data['hosp_patients_per_million']/(covid_data['hospital_beds_per_thousand']*1000))
    covid_data.drop(columns=['hosp_patients_per_million','hospital_beds_per_thousand'], inplace= True) #drop uncessary columns
    
    # Preprocess the data for each country individually
    country_data_dict = dict(tuple(covid_data.groupby("location"))) # create a dictionary of dataframes with an entry for each country
    
    for country in full_name_countries:  # For each country:
        country_data_dict[country].fillna(method='ffill', inplace=True)  # Fill in missing values with the most recent previous value
        country_data_dict[country].fillna(0, inplace=True)  # If there is no previous value, fill in with 0
        country_data_dict[country].rename(columns={'date': 'daydate',  # Rename columns to include country name
                                                   'new_cases_smoothed_per_million': f'{country}_new_cases_smoothed_per_million', 
                                                   'new_deaths_smoothed_per_million': f'{country}_new_deaths_smoothed_per_million', 
                                                   'people_vaccinated_per_hundred':f'{country}_people_vaccinated_per_hundred', 
                                                   'weekly_icu_admissions_per_million':f'{country}_weekly_icu_admissions_per_million',
                                                   'new_tests_smoothed_per_thousand':f'{country}_new_tests_smoothed_per_thousand',
                                                   'median_age':f'{country}_median_age',
                                                   'gdp_per_capita':f'{country}_gdp_per_capita',
                                                   'human_development_index':f'{country}_human_development_index',
                                                   #'hosp_patients_per_million':f'{country}_hosp_patients_per_million',
                                                   #'hospital_beds_per_thousand':f'{country}_hospital_beds_per_thousand',
                                                   'hosp_beds_capacity_per_million':f'{country}_hosp_beds_capacity_per_million'},
                                          inplace=True)
        
        country_data_dict[country].drop('location', axis=1, inplace=True)   # drop the now superfluous country name column
        date_df = date_df.merge(country_data_dict[country], on = 'daydate', how = 'left')  # merge country dataframes into the day-by-day skeleton
        
    date_df.loc[date_df['daydate'] < min(covid_data['date']) , date_df.columns != 'daydate'] = 0  # Set values before the start of the covid dataset to zero
    
    return date_df


# ## 2.2 Mobility Functions

# In[5]:


def preprocess_mobility_data(country_list=defines.country_list):
    '''
    Downloads and preprocesses data for each country in country_list (by default taken from defines.py)
    Data from before the start of the source data is filled in with the nearest available value. 
    Values past the end of the source data set remain as NaN (intended to be filled in when the data is merged)
    '''
    
    # Define which columns to download
    req_cols = ['date', 'country_region', 'retail_and_recreation_percent_change_from_baseline',
                                            'grocery_and_pharmacy_percent_change_from_baseline',
                                            'parks_percent_change_from_baseline',
                                            'transit_stations_percent_change_from_baseline',
                                            'workplaces_percent_change_from_baseline',
                                            'residential_percent_change_from_baseline']

    # Download raw data from data source
    raw_mobility_data = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', usecols = req_cols)  
    
    # Convert country abbreviations into full country names (DE -> Germany)
    full_name_countries = []
    for country in country_list:
        full_name_countries.append(pycountry.countries.get(alpha_2=country).name)
    full_name_countries
    
    # Select only countries from country list 
    mobility_df = raw_mobility_data[raw_mobility_data['country_region'].isin(full_name_countries)]
    del raw_mobility_data # delete raw data to free up memmory
    
    mobility_df['date'] = pd.to_datetime(mobility_df['date'])   
    

    date_df = create_dateframe() # create day-by-day dataframe

    # Preprocess the data for each country individually
    country_data_dict = dict(tuple(mobility_df.groupby("country_region"))) # create a dictionary of dataframes with an entry for each country
    
    for country in full_name_countries:  # For each country:
        country_data_dict[country] = country_data_dict[country].groupby('date').mean() # Take the mean value for each day (averaging out regional differences)
        country_data_dict[country] = country_data_dict[country].reset_index()
        country_data_dict[country].fillna(method='ffill', inplace=True)  # Fill in missing values with the most recent previous value
        country_data_dict[country].fillna(0, inplace=True)  # If there is no previous value, fill in with 0

         # Rename columns to include country name
        country_data_dict[country].rename(columns={'date': 'daydate', 
                                                    'retail_and_recreation_percent_change_from_baseline':f'{country}_retail_and_recreation_percent_change_from_baseline',
                                                    'grocery_and_pharmacy_percent_change_from_baseline':f'{country}_grocery_and_pharmacy_percent_change_from_baseline',
                                                    'parks_percent_change_from_baseline':f'{country}_parks_percent_change_from_baseline',
                                                    'transit_stations_percent_change_from_baseline':f'{country}_transit_stations_percent_change_from_baseline',
                                                    'workplaces_percent_change_from_baseline':f'{country}_workplaces_percent_change_from_baseline',
                                                    'residential_percent_change_from_baseline':f'{country}_residential_percent_change_from_baseline'},
                                            inplace=True)
        date_df = date_df.merge(country_data_dict[country], on = 'daydate', how = 'left')  # merge country dataframes into the day-by-day skeleton

    date_df = date_df.fillna(method='ffill')  # fill in values from before start of data with nearest available data
    
    return date_df


# ## 2.3 Stringency Functions

# In[6]:


def preprocess_stringency_data(country_list=defines.country_list):
    '''
    Takes a list of two letter country abbreviations (by default from defines.py) and a dataframe containing day-by-day data for the covid stringency index.
    If any previous data exists, missing data is filled in with the most recent value. If there is no previous data missing values are assumed to be zero.
    Values past the end of the source data set remain as NaN (intended to be filled in when the data is merged)
    '''
    
    # Download raw data from data source
    raw_stringency_data = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv')

    # Convert country abbreviations into full country names (DE -> Germany)
    full_name_countries = []
    for country in country_list:
        full_name_countries.append(pycountry.countries.get(alpha_2=country).name)
    full_name_countries
    
    # Select only countries in country list
    stringency_df = raw_stringency_data[raw_stringency_data['country_name'].isin(full_name_countries)]
    
    # Convert data from wide to long format
    date_columns = list(stringency_df.columns[7:]) # list all date columns 
    stringency_pivot = pd.melt(stringency_df, id_vars= 'country_name', value_vars= date_columns, var_name= 'daydate', value_name= 'Stringency_Index') # convert to long format
    stringency_pivot['daydate'] = pd.to_datetime(stringency_pivot['daydate'])   

    # Preprocess the data for each country individually
    country_data_dict = dict(tuple(stringency_pivot.groupby("country_name"))) # create a dictionary of dataframes with an entry for each country

    date_df = create_dateframe() # create day-by-day dataframe

    for country in full_name_countries:  # For each country:
        country_data_dict[country].fillna(method='ffill', inplace=True)  # Fill in missing values with the most recent previous value
        country_data_dict[country].fillna(0, inplace=True)  # If there is no previous value, fill in with 0

        # Drop superfluous location column
        country_data_dict[country] = country_data_dict[country].drop('country_name', axis = 1)

         # Rename columns to include country name
        country_data_dict[country].rename(columns={'daydate': 'daydate', 
                                                   'Stringency_Index': f'{country}_Stringency_Index'}, 
                                            inplace=True)
        
        # Merge country dataframes into the day-by-day skeleton
        date_df = date_df.merge(country_data_dict[country], on = 'daydate', how = 'left')  

    date_df = date_df.fillna(method='ffill')  # fill in values from before start of data with nearest available data

    return date_df


# ## 2.4 General Functions

# In[7]:


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

def preprocess_all_covid_data(country_list=defines.country_list):
    covid_df = preprocess_covid_data(country_list)
    mobility_df = preprocess_mobility_data(country_list)
    stringency_df = preprocess_stringency_data(country_list)
    covid_df = covid_df.merge(mobility_df, on='daydate', how='left')
    covid_df = covid_df.merge(stringency_df, on='daydate', how='left')
    
    return covid_df


# # 3. Preprocessing

# In[8]:


merged_covid_data = preprocess_all_covid_data()
merged_covid_data.fillna(0, inplace=True)


# In[9]:


#merged_covid_data


# # 4. Read to database

# In[10]:


# from: https://stackoverflow.com/questions/58203973/pandas-unable-to-write-to-postgres-db-throws-keyerror-select-name-from-sqlit
# RDBMs including Postgres must use an SQLAlchemy connection for this method to create structures and append data. 
# However, read_sql does not require SQLAlchemy since it does not make persistent changes.

# create sqlalchemy connection
#from sqlalchemy import create_engine

#params = config()

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"))


# In[11]:


# read dataframe to database schema transformed and replace entire table if existing already
merged_covid_data.to_sql("external_covid_data", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")


# In[12]:


engine.dispose()


# In[13]:


print(params)

