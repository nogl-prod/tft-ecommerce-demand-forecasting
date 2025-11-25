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


engines = {
    "dev" : get_db_connection(dbtype="dev", client_name=args.client_name),
    "prod" : get_db_connection(dbtype="prod", client_name=args.client_name)
}

#mergable_id = best_mergable_id(args.client_name, "plan_data")
mergable_id = "product_category"

productsxdays_sceleton = import_data_AWSRDS(schema="product",table="productsxdays_sceleton",engine=engines["dev"])
# remove shopify prefix from the columns if col contains shopify
productsxdays_sceleton.columns = [col.replace("shopify_","") if "shopify_" in col else col for col in productsxdays_sceleton.columns]
# make first row as column names
weekly_plan_data = pd.read_excel("/opt/ml/processing/input/Data Source Transformations/MSC/Plan Data/"+args.client_name+"_Plan_Data_Weekly.xlsx",header=0)

# make first row as column names
weekly_plan_data.columns = weekly_plan_data.iloc[0]
weekly_plan_data.drop(index=0, axis=0, inplace=True)

def create_category_number_mapping(productsxdays_sceleton):
    category_mapping = productsxdays_sceleton.groupby(['product_category','product_category_number']).count().reset_index()
    category_mapping.drop(columns=["daydate", "variant_sku","variant_id"], inplace=True)
    return category_mapping


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
    # convert daydate to datetime
    skeleton["daydate"] = pd.to_datetime(skeleton["daydate"])
    skeleton['iso_week_id'] = skeleton['daydate'].apply(lambda row: f'{row.isocalendar()[0]}-{row.isocalendar()[1]}') 
    
    # Merge daily plan data onto product x day skeleton via iso_week_id
    # rename product_category to mergable_id
    # Merge daily plan data onto product x day skeleton via iso_week_id
    plan_data_df.drop(columns=["daydate", "product_category"], inplace=True)
    filled_skeleton = skeleton.merge(plan_data_df, on = ['iso_week_id', 'product_category_number'], how = 'left')
    
    # Fill all NAs with zero
    filled_skeleton = filled_skeleton.fillna(0)
    # Drop superfluous columns
    filled_skeleton = filled_skeleton[['daydate', "nogl_id", 'variant_sku','variant_id', 'product_category_number',
                                        "amazon_variant_sku", "amazon_variant_asin", "amazon_product_category_number",
                                       'daily_FB_Budget', 'daily_GoogleLeads_Budget','daily_Other_Marketing_Budget', 
                                       'daily_Product_Sales_Target', 'daily_num_planned_klaviyo_campaigns', 'daily_planned_klaviyo_grossreach']]
    
    return filled_skeleton


daily_plan_data = preprocess_plan_data(weekly_plan_data, productsxdays_sceleton)
complete_plan_data = merge_plan_data_to_product_x_day_skeleton(daily_plan_data, productsxdays_sceleton)
complete_plan_data.fillna(0, inplace=True)
complete_plan_data.rename(columns = {"variant_sku" : "shopify_variant_sku",
                                     "variant_id" : "shopify_variant_id",
                                     "product_category_number" : "shopify_product_category_number"}, inplace = True)

from sqlalchemy import create_engine, text
# Assuming the rest of your code above this line is importing necessary modules and defining other variables...
t = Timer("Export")
# rename nogl_id to variant_id
complete_plan_data.rename(columns={"nogl_id": "variant_id"}, inplace=True)
complete_plan_data.to_sql('plan_data', con=engines["dev"], schema="product", if_exists='replace', index=False) # drops old table and creates new empty table
t.end()