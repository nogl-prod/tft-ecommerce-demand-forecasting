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

mergable_id = get_best_mergable_id(args.client_name, "klaviyo")
klaviyo_events_raw = import_data_AWSRDS(schema="klaviyo",table="events_scd",engine=engines["prod"])
klaviyo_campaigns_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns",engine=engines["prod"])
klaviyo_campaigns_lists_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns_lists",engine=engines["prod"])
klaviyo_lists_raw = import_data_AWSRDS(schema="klaviyo",table="lists",engine=engines["prod"])
# Load Product ID Data
klaviyo_productsxdays_sceleton = import_data_AWSRDS(schema="product",table="productsxdays_sceleton",engine=engines["dev"])
klaviyo_events_raw["event_properties"]= klaviyo_events_raw["event_properties"].astype("string")


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
    klaviyo_productsxdays_sceleton = product_X_day_skeleton # [['variant_sku', 'daydate', 'variant_id']]  # Drop all non-index columns 
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
    skuXdate_df.rename(columns = {'datetime' : 'daydate', 'sku_id' : mergable_id}, inplace = True) 
    skuXdate_df['daydate'] = pd.to_datetime(skuXdate_df['daydate'])
    print(skuXdate_df.columns)
    
    # Clean event data
    skuXdate_df = skuXdate_df.fillna(0)
    skuXdate_df = skuXdate_df.drop_duplicates()

    # Fill product x day Skeleton with klaviyo data
    klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.merge(skuXdate_df, on = ['daydate', mergable_id], how = 'left')
    klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.fillna(0) # fill all created NANs with zero

    return klaviyo_productsxdays_sceleton



# Create the current klaviyo from the airbyte datasets 
filled_productsxdays_sceleton = klaviyo_create_productxday_skeleton(klaviyo_events_raw, klaviyo_campaigns_raw, klaviyo_lists_raw, klaviyo_campaigns_lists_raw, klaviyo_productsxdays_sceleton)

complete_klaviyo_data = filled_productsxdays_sceleton.copy()

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

from sqlalchemy import create_engine, text
# Assuming the rest of your code above this line is importing necessary modules and defining other variables...
t = Timer("Export")
# rename nogl id to variant_id
complete_klaviyo_data.rename(columns={"nogl_id": "variant_id"}, inplace=True)
complete_klaviyo_data.to_sql('msc_klaviyo', con=engines["dev"], schema="transformed", if_exists='replace', index=False) # drops old table and creates new empty table
t.end()