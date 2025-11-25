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

################################################################################################################
products = import_data_AWSRDS(schema="product",table="product_info",engine=engines["dev"])
# get all products from products where shopify_variant_barcode or sku is not null
shopify_products = products[products["shopify_variant_barcode"].notnull() | products["shopify_variant_sku"].notnull()]
shopify_products = shopify_products[["variant_id", "shopify_variant_id", "shopify_variant_sku"]]
# remove shopify prefix from the columns names
shopify_sales = import_data_AWSRDS(schema="transformed",table="shopify_sales",engine=engines["prod"])
drop_cols = ['variant_grams', 'variant_RRP',
'variant_taxable', 'variant_position', 'variant_created_at',
'variant_inventory_item_id', 'variant_requires_shipping',
'variant_inventory_management_used',
'product_published_at', 'product_published_scope',
'variant_max_updated_at']
# drop cols from sales
shopify_sales = shopify_sales.drop(drop_cols,axis=1)
shopify_sales = shopify_sales.rename(columns={"variant_id":"shopify_variant_id", "variant_sku":"shopify_variant_sku"})
shopify_sales = shopify_sales.merge(shopify_products[["variant_id", "shopify_variant_id", "shopify_variant_sku"]], on=["shopify_variant_id", "shopify_variant_sku"], how="left")

from sqlalchemy import create_engine, text
# Assuming the rest of your code above this line is importing necessary modules and defining other variables...
shopify_sales.to_sql('msc_shopify_sales', con=engines["dev"], schema="transformed", if_exists='replace', index=False) # drops old table and creates new empty table
shopify_sales.to_sql('historical_sales', con=engines["dev"], schema="sc_shopify", if_exists='replace', index=False) 
################################################################################################################