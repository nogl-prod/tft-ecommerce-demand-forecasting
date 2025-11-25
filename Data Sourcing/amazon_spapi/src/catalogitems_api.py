# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import CatalogItems
from sp_api.base import Marketplaces
from sp_api.base import SellingApiRequestThrottledException

# internal imports
from .utils import *
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def enrich_dataframe_with_ean(df, marketplace, credentials):
    """
    Enriches a dataframe with the EAN for each ASIN.
    
    Parameters:
    - df: The input dataframe. It must contain a column named 'asin'.
    - marketplace: The marketplace object.
    - credentials: The credentials for the CatalogItems API.
    
    Returns:
    - A new dataframe with an additional 'variant_barcode' column.
    """
    
    # Initialize the endpoint within the function
    catalogitems_endpoint = CatalogItems(credentials=credentials, marketplace=marketplace)

    def get_all_eans(asin, marketplace):
        retry_attempts = 5
        retry_duration = 2  # seconds, you can adjust this duration based on your preference

        for attempt in range(retry_attempts):
            try:
                response = catalogitems_endpoint.get_catalog_item(asin=asin, 
                                                                  marketplaceIds=marketplace.marketplace_id, 
                                                                  includedData=["identifiers"]).payload

                # Extract EAN from the response
                for identifier_group in response.get('identifiers', []):
                    for identifier_item in identifier_group.get('identifiers', []):
                        if identifier_item['identifierType'] == 'EAN':
                            return identifier_item['identifier']

            except Exception as e:  # Adjust this to catch the specific exception for rate limiting
                if 'Rate exceeded' in str(e):  # Adjust this string to match the exact rate limit error message
                    time.sleep(retry_duration)
                    continue
                else:
                    print(f"Error for ASIN {asin}: {e}")
                    return None
        print(f"Failed to get EAN for ASIN {asin} after {retry_attempts} attempts.")
        return None

    # Copy the dataframe to avoid modifying the original one
    enriched_df = df.copy()
    enriched_df['variant_barcode'] = enriched_df['asin'].apply(lambda x: get_all_eans(x, marketplace))
    
    return enriched_df
