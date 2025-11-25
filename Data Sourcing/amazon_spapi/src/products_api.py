# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import Products
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType
from sp_api.base import SellingApiRequestThrottledException

# internal imports
from .utils import *
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def get_product_pricing_and_offer_data(asin, item_condition, credentials, marketplace):
    """
    This function returns a DataFrame with pricing information for a list of SKUs. 
    It makes an API call to Amazon SP-API and processes the response into a flat DataFrame structure.

    Parameters:
    seller_id (str): The seller's id.
    sku_list (list): List of SKUs for which pricing information is requested.

    Returns:
    df (pd.DataFrame): DataFrame with pricing information.
    """

    products_api = Products(credentials=credentials, marketplace=marketplace)

    product_pricing = products_api.get_product_pricing_for_asins(asin_list=[asin], item_condition=item_condition, marketplaceIds=[marketplace.marketplace_id]) # not only head(20) but all products
    product_pricing_normalized = denest_response_json_to_df(product_pricing.payload)

    item_offers = products_api.get_item_offers(asin=asin, item_condition=item_condition, marketplace_id=marketplace.marketplace_id)
    item_offers_normalized = denest_response_json_to_df(item_offers.payload)

    competitive_pricing = products_api.get_competitive_pricing_for_asins(asin_list=[asin], marketplaceIds=[marketplace.marketplace_id]) # not only head(20) but all products
    competitive_pricing_normalized = denest_response_json_to_df(competitive_pricing.payload)
    
    product_pricing_normalized, item_offers_normalized = drop_overlapping_columns(product_pricing_normalized, item_offers_normalized, columns_to_keep=["ASIN"])
    product_pricing_and_offer_data = product_pricing_normalized.merge(item_offers_normalized, on="ASIN", how="left")

    product_pricing_and_offer_data, competitive_pricing_normalized = drop_overlapping_columns(product_pricing_and_offer_data, competitive_pricing_normalized, columns_to_keep=["ASIN"])
    product_pricing_and_offer_data = product_pricing_and_offer_data.merge(competitive_pricing_normalized, on="ASIN", how="left")

    return product_pricing_and_offer_data

def get_all_product_pricing_and_offer_data(amazon_products, marketplace: str, credentials, products_mapping, asin_identifier = "asin", item_condition_identifier = "item_condition_number"):
    """
    This function retrieves pricing and offer data for each product in a given DataFrame from the Amazon SP-API. 
    If it encounters a SellingApiRequestThrottledException indicating the API request quota has been exceeded, 
    it waits for 1 second before retrying. After data retrieval, it drops overlapping columns and merges the data back 
    into the original DataFrame.

    Parameters:
    amazon_products (pd.DataFrame): DataFrame containing product data, including ASIN and item condition identifiers.
    marketplace (str): The marketplace to get product pricing and offer data from.
    credentials: Amazon SP-API credentials.
    products_mapping (dict): A dictionary mapping current column names to desired column names.
    asin_identifier (str, optional): The column name for ASIN in 'amazon_products'. Defaults to "asin".
    item_condition_identifier (str, optional): The column name for item condition in 'amazon_products'. Defaults to "item_condition_number".

    Returns:
    pd.DataFrame: The original DataFrame merged with the retrieved product pricing and offer data.
    """
    product_pricing_and_offers = pd.DataFrame()

    for _, row in amazon_products[[asin_identifier, item_condition_identifier]].iterrows():
        asin = row[asin_identifier]
        item_condition_type = map_item_condition(row[item_condition_identifier], case="capital")

        while True:
            try:
                asin_product_pricing_and_offers = get_product_pricing_and_offer_data(asin, item_condition_type, marketplace=marketplace, credentials=credentials)
                break  # If successful, exit the while loop
            except SellingApiRequestThrottledException:  # Assuming this is the correct Exception to catch
                print(f"Quota exceeded for ASIN: {asin}, sleeping for 1 second.")
                time.sleep(1)

        product_pricing_and_offers = pd.concat([product_pricing_and_offers, asin_product_pricing_and_offers], axis=0)

    product_pricing_and_offers.rename(columns=products_mapping, inplace=True)
    product_pricing_and_offers = product_pricing_and_offers.T.drop_duplicates().T # remove duplicate columns
    
    amazon_products, product_pricing_and_offers = drop_overlapping_columns(amazon_products, product_pricing_and_offers, columns_to_keep=[asin_identifier])
    amazon_products = pd.merge(amazon_products, product_pricing_and_offers, on=asin_identifier, how="left")
    
    return amazon_products