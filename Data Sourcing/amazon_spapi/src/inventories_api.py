# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import Inventories
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType

# internal imports
from .utils import *
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def get_inventory(seller_id: str, seller_sku_list: list, marketplace: str, credentials, marketplace_ids: list = None) -> pd.DataFrame:
    """
    Retrieves inventory summary for a specific seller and list of SKUs from Amazon SP-API.

    Parameters:
    seller_id (str): The seller's ID.
    seller_sku_list (list): The list of SKUs for which inventory summary is requested.
    marketplace (str): The marketplace ID.
    credentials: The credentials required for authentication.
    marketplace_ids (list): The list of marketplace IDs.

    Returns:
    inventories_normalized (pd.DataFrame): The normalized DataFrame containing the inventory summary.
    """
    if marketplace_ids == None:
        marketplace_ids = [marketplace.marketplace_id]

    inventory_api = Inventories(credentials=credentials, marketplace=marketplace)
    
    # Get the inventory summary for the specified SKUs
    response = inventory_api.get_inventory_summary_marketplace(
        sellerId=seller_id,
        sellerSkus=seller_sku_list,
        details=True,
        marketplaceIds=marketplace_ids
    )
    
    # Normalize the response payload into a DataFrame
    inventories_normalized = denest_response_json_to_df(response.payload["inventorySummaries"])

    return inventories_normalized

def get_all_inventory_data(amazon_products, seller_id: str, marketplace: str, credentials, inventory_mapping, sku_identifier = "sku", sublist_size = 50):
    """
    Fetches all the inventory data for a list of SKU identifiers from Amazon products, and merges the inventory
    data into the original Amazon products DataFrame. Fetching is performed in chunks of size `sublist_size`.

    Parameters
    ----------
    amazon_products : pandas.DataFrame
        DataFrame containing Amazon product information.
    seller_id : str
        The ID of the seller for which inventory information is to be fetched.
    marketplace : str
        The marketplace from which inventory information is to be fetched.
    credentials
        Credentials required to access the marketplace data.
    inventory_mapping : dict
        A dictionary for mapping inventory column names.
    sku_identifier : str, optional
        The SKU identifier column name in the amazon_products DataFrame, default is "sku".
    sublist_size : int, optional
        The number of SKU identifiers to fetch inventory information for in each chunk, default is 50.

    Returns
    -------
    amazon_products : pandas.DataFrame
        The updated DataFrame with all inventory data merged in for the specified SKU identifiers.

    Notes
    -----
    The function uses an externally defined function `get_inventory`, which is assumed to take a seller ID,
    a list of SKUs, a marketplace, and credentials, and returns a DataFrame of inventory data for those SKUs.
    """
    inventories = pd.DataFrame()
    all_skus = amazon_products[sku_identifier].tolist()

    for i in range(0, len(all_skus), sublist_size):
        sublist = all_skus[i:i+sublist_size]
        sub_inventories = get_inventory(seller_id, seller_sku_list = sublist, marketplace=marketplace, credentials=credentials)
        inventories = pd.concat([inventories, sub_inventories], axis=0)
        time.sleep(1)

    inventories.rename(columns=inventory_mapping, inplace=True)

    amazon_products, inventories = drop_overlapping_columns(amazon_products, inventories, columns_to_keep=[sku_identifier])

    amazon_products = pd.merge(amazon_products, inventories, on=sku_identifier, how="left")
    return amazon_products