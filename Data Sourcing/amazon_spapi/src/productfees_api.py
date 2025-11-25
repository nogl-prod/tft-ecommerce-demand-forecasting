# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import ProductFees
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType
from sp_api.base import SellingApiRequestThrottledException

# internal imports
from .utils import *
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def get_product_fees(asin: str, price: float, marketplace: str, credentials, currency: str = "EUR") -> pd.DataFrame:
    """
    Retrieves the product fees estimate for a specific ASIN at a given price in a marketplace.

    Parameters:
        asin (str): The ASIN (Amazon Standard Identification Number) of the product.
        price (float): The price of the product for which to estimate the fees.
        marketplace (str): The marketplace where the product is listed.
        credentials: The credentials required to access the Amazon SP-API.
        currency (str, optional): The currency code to use for the fees estimate. Defaults to "EUR".

    Returns:
        pd.DataFrame: A DataFrame containing the normalized product fees estimate.

    Raises:
        Any exceptions raised by the underlying API calls.

    Example:
        product_fees = get_product_fees(
            asin="B12345",
            price=19.99,
            marketplace="US",
            credentials=my_credentials,
            currency="USD"
        )
    """

    product_fees_api = ProductFees(credentials=credentials, marketplace=marketplace)
    
    # Get the inventory summary for the specified SKU
    response = product_fees_api.get_product_fees_estimate_for_asin(
        asin=asin,
        price=price,
        currency=currency
    )
    
    # Normalize the response payload into a DataFrame
    product_fees_normalized = denest_response_json_to_df(response.payload)

    json_data = product_fees_normalized["FeesEstimateResult"].iloc[2]

    # Create an empty dictionary to store the normalized data
    normalized_data = {}

    # Extract the currency code from the top nesting level
    normalized_data['CurrencyCode'] = json_data['TotalFeesEstimate']['CurrencyCode']

    # Extract the total fees estimate amount
    normalized_data['TotalFeesEstimateAmount'] = json_data['TotalFeesEstimate']['Amount']

    # Extract the fee amounts for each fee type
    for fee_detail in json_data['FeeDetailList']:
        fee_type = fee_detail['FeeType']
        fee_amount = fee_detail['FeeAmount']['Amount']
        normalized_data[f'{fee_type}Amount'] = fee_amount

    normalized_data = pd.DataFrame(normalized_data, index=[0])

    normalized_data["asin"] = asin

    return normalized_data

def get_all_product_fees(amazon_products, marketplace: str, credentials, productfees_mapping: dict, asin_identifier = "asin", price_identifier = "price"):
    """
    Fetches and aggregates the fees for a set of products from the Amazon Selling Partner API.

    Parameters
    ----------
    amazon_products : pd.DataFrame
        DataFrame containing information about Amazon products. It must contain columns specified 
        by asin_identifier and price_identifier parameters.

    marketplace : str
        A string indicating the marketplace for the API call.

    credentials : Object
        An object containing Amazon API credentials.

    productfees_mapping : dict
        A dictionary to map the columns of the product_fees DataFrame to new names.

    asin_identifier : str, optional
        The name of the column in amazon_products DataFrame containing the ASINs. 
        Default is "asin".

    price_identifier : str, optional
        The name of the column in amazon_products DataFrame containing the prices. 
        Default is "price".

    Returns
    -------
    amazon_products : pd.DataFrame
        Returns the original DataFrame (amazon_products) with additional columns containing product fee 
        information. Columns with the same name in amazon_products and product_fees are dropped from 
        product_fees before merging, except for the column specified by asin_identifier.

    Raises
    ------
    SellingApiRequestThrottledException
        This exception is raised when the API call rate limit has been exceeded. When this happens, 
        the function sleeps for 1 second and then tries to fetch the fees for the current product again.

    Notes
    -----
    The function makes a series of API calls to fetch the product fees for each product in the 
    amazon_products DataFrame. The fees are then added to a new DataFrame (product_fees). This new 
    DataFrame is merged with the original amazon_products DataFrame, and the merged DataFrame is returned.
    """
    product_fees = pd.DataFrame()

    for _, row in amazon_products[[asin_identifier, price_identifier]].iterrows():
        asin = row[asin_identifier]
        price = row[price_identifier]
        if pd.isna(price) == True:
            continue            

        while True:
            try:
                asin_product_fees = get_product_fees(asin, price, marketplace=marketplace, credentials=credentials)
                break  # If successful, exit the while loop
            except SellingApiRequestThrottledException:  # Assuming this is the correct Exception to catch
                print(f"Quota exceeded for ASIN: {asin}, sleeping for 1 second.")
                time.sleep(1)

        product_fees = pd.concat([product_fees, asin_product_fees], axis=0)

    product_fees.rename(columns=productfees_mapping, inplace=True)
    
    amazon_products, product_fees = drop_overlapping_columns(amazon_products, product_fees, columns_to_keep=[asin_identifier])
    amazon_products = pd.merge(amazon_products, product_fees, on=asin_identifier, how="left")
    
    return amazon_products