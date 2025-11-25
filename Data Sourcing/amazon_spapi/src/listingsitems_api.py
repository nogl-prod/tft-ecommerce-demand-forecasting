# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import ListingsItems
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType
from sp_api.base.exceptions import SellingApiNotFoundException

# internal imports
from .utils import *
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def get_listings_item(seller_id: str, sku: str, marketplace: str, credentials, marketplace_ids: list = None) -> pd.DataFrame:
    """
    Retrieves listing items for a specific seller and SKU from Amazon SP-API.

    Parameters:
    seller_id (str): The seller's ID.
    sku (str): The SKU of the listing item.
    marketplace (str): The marketplace ID.
    marketplace_ids (list): The list of marketplace IDs.
    credentials: The credentials required for authentication.

    Returns:
    listing_normalized (pd.DataFrame): The normalized DataFrame containing the listing items.
    """
    if marketplace_ids == None:
        marketplace_ids = [marketplace.marketplace_id]
        
    listings_api = ListingsItems(credentials=credentials, marketplace=marketplace)
    
    # Get the listings item
    response = listings_api.get_listings_item(sellerId=seller_id, sku=sku, marketplaceIds=marketplace_ids)
    
    # Normalize the response payload into a DataFrame
    listing_normalized = denest_response_json_to_df(response.payload)

    return listing_normalized


def get_all_listing_items(amazon_products, seller_id: str, marketplace: str, credentials, listing_items_mapping, sku_identifier):
    """
    Extracts all the listing items for the specified SKU identifiers from Amazon products,
    merges them into the original Amazon products DataFrame, and returns the updated DataFrame.

    Parameters
    ----------
    amazon_products : pandas.DataFrame
        DataFrame containing Amazon product information.
    sku_identifier : str, optional
        The SKU identifier column name in the amazon_products DataFrame, default is "sku".

    Returns
    -------
    amazon_products : pandas.DataFrame
        The updated DataFrame with all listing items merged in for the specified SKU identifiers.

    Notes
    -----
    The function uses an externally defined function `get_listings_item` and `drop_overlapping_columns`.
    `get_listings_item` is assumed to take a seller ID, SKU, marketplace, and credentials,
    and returns a DataFrame of listings for that SKU.
    `drop_overlapping_columns` is used to avoid column name conflicts when merging the listings into the original DataFrame.
    """
    # listings = pd.DataFrame()
    # for sku in amazon_products[sku_identifier]:
    #     sku_listing = get_listings_item(seller_id, sku, marketplace=marketplace, credentials=credentials)
    #     listings = pd.concat([listings, sku_listing], axis=0)

    # listings.rename(columns=listing_items_mapping, inplace=True)
    
    listings = pd.DataFrame()

    for sku in amazon_products[sku_identifier]:
        try:
            sku_listing = get_listings_item(seller_id, sku, marketplace=marketplace, credentials=credentials)
            listings = pd.concat([listings, sku_listing], axis=0)
        except SellingApiNotFoundException:
            print(f"SKU '{sku}' not found in marketplace {marketplace.marketplace_id}. Skipping...")
        except Exception as e:
            print(f"Error fetching details for SKU '{sku}': {e}. Skipping...")

    listings.rename(columns=listing_items_mapping, inplace=True)

    amazon_products, listings = drop_overlapping_columns(amazon_products, listings, columns_to_keep=[sku_identifier])

    amazon_products = amazon_products.merge(listings, how="left", on=sku_identifier)

    return amazon_products