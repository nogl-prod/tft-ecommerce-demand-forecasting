# external imports
import pandas as pd
import time
import os, sys, pathlib

# SP-API import
from sp_api.base import Marketplaces
from sp_api.api import Orders
from sp_api.util import throttle_retry, load_all_pages

# internal imports
from .database_connection_operators import *
from .utils import *

src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def get_orders(start_date, credentials, column_mapping, requests_per_second=0.5):
    """
    This function retrieves order and order item details from Amazon SP-API for a given start date, credentials, and client name. It handles rate limits by sleeping between requests.

    Parameters:
        start_date (str): The start date for which to retrieve orders. The date should be in ISO 8601 format (YYYY-MM-DD).
        credentials (dict): The credentials to use for the SP-API. This should be a dictionary containing the necessary authentication details.
        column_mapping (dict): A dictionary mapping the original column names to the new column names. The keys should be the original column names and the values should be the new column names.
        requests_per_second (float, optional): The number of requests to make per second. This is used to handle rate limits. Default is 0.5.

    Returns:
        df (pandas.DataFrame): A DataFrame containing the order and order item details. The DataFrame will have one row per order item and columns for both order and item details. The column names will be prefixed with "order_" or "lineitem_" to indicate whether they contain order or item details, respectively. The column names will be renamed according to the provided column_mapping.
    """
    orders_endpoint = Orders(credentials=credentials, marketplace=Marketplaces.DE)

    @throttle_retry()
    @load_all_pages()
    def load_all_orders(orders_endpoint, **kwargs):
        """
        a generator function to return all pages, obtained by NextToken
        """
        return orders_endpoint.get_orders(**kwargs)
    
    @throttle_retry()
    @load_all_pages()
    def load_order_items(orders_endpoint, order_id, **kwargs):
        """
        a generator function to return all items for a specific order, obtained by NextToken
        """
        return orders_endpoint.get_order_items(order_id, **kwargs)

    # Initialize an empty list to store all order items
    all_order_items = []

    for page in load_all_orders(orders_endpoint = orders_endpoint, LastUpdatedAfter=start_date):
        for order in page.payload.get('Orders'):
            order_id = order.get('AmazonOrderId')
            time.sleep(((1/requests_per_second)+0.05))
            for item_page in load_order_items(orders_endpoint, order_id):
                # For each order item, create a new DataFrame that combines the order and item details
                for item in item_page.payload.get('OrderItems', []):
                    order_df = pd.json_normalize(order)  # Flatten the order details
                    order_df.columns = ["order_" + col for col in order_df.columns]  # Add "order_" prefix to column names
                    item_df = pd.json_normalize(item)  # Flatten the item details
                    item_df.columns = ["lineitem_" + col for col in item_df.columns]  # Add "lineitem_" prefix to column names
                    order_item_df = pd.concat([order_df, item_df], axis=1)  # Concatenate the order and item DataFrames
                    all_order_items.append(order_item_df)

    # Concatenate all the order item DataFrames into a single DataFrame
    df = pd.concat(all_order_items, ignore_index=True)

    # Only keep keys that exist in df.columns
    keys = list(column_mapping.keys())
    existing_keys = [key for key in keys if key in df.columns]
    df = df[existing_keys] # only keep selected features

    # rename with dictionary
    df.rename(columns=column_mapping, inplace=True)

    return df

def raw_orders_sync(start_date, credentials, client_name, column_mapping, params, unique_ids = ["order_amazon_order_id", "lineitem_order_item_id"], table_name="order_line_items"):
    amazon_exists = check_table_exists("amazon", table_name, params)

    if amazon_exists == False:
        print(f"Downloading Amazon Orders Data for the first time for client {client_name} from starting date {start_date}.")

        orders_df = get_orders(start_date, credentials, column_mapping)
        
        create_schema("amazon", params)

        # Adjust the data types of the unique identifier columns to text
        for uid in unique_ids:
            orders_df[uid] = orders_df[uid].astype(str)

        save_to_db(orders_df, "amazon", table_name, params)

        return orders_df # DELETE LATER
    else:
        # update only rows that have been changed in the previous timeframe

        # get max date from existing data
        last_order_update_date = get_max_value("amazon", table_name, "order_last_update_date", params)

        print(f"Updating old and downloading new Amazon Orders Data for client {client_name} with last update after {last_order_update_date}.")

        # run get_list_of_orders again
        new_orders_df = get_orders(last_order_update_date, credentials, column_mapping)

        # update rows / add new rows
        update_table_from_dataframe(new_orders_df, table_name, "amazon", params, unique_ids)
        
        return new_orders_df # DELETE LATER