# Imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import time
from typing import Optional
from utils import *
import logging
import datetime
import argparse
import time
from Support_Functions import *

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

# parser = argparse.ArgumentParser(description="This script creates product info table in database with unique nogl id.")

# CLIENT_NAME = "dogs-n-tiger"
# TRANSFORMATION = "google_ads"

# parser.add_argument(
#     "--client_name",
#     type=str,
#     default=CLIENT_NAME,
#     help="Client name",
# )

# parser.add_argument(
#     "--transformation",
#     type=str,
#     default=TRANSFORMATION,
#     help="transformation",
# )


# args = parser.parse_args()
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



def get_best_mergable_id(client_name, transformation):

    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }

    products_dicts = {}
    with log_time("Importing data from AWS RDS"):
        

        linkage_df = import_data_AWSRDS(schema="product", table="product_info", engine=engines["dev"])
        # TODO : update this later on
        # rename variant_id to nogl_id
        linkage_df.rename(columns={"variant_id": "nogl_id"}, inplace=True)
        # create a new column with shopify_variant_id and name variant_id
        linkage_df["variant_id"] = linkage_df["shopify_variant_id"]


        sceleton_dict = {}
        sceleton = pd.DataFrame()
        id_count = 0
        for channel in channel_names:
            # TODO : change transformed schema name later to sc_channel name
            sceleton_dict[channel] = import_data_AWSRDS(schema="sc_"+channel, table=channel+"_productsxdays_sceleton", engine=engines["dev"])
            if id_count == 0:
                sceleton = import_data_AWSRDS(schema="sc_"+channel, table=channel+"_productsxdays_sceleton", engine=engines["dev"])
                id_count += 1
            else:
                sceleton = sceleton.merge(sceleton_dict[channel],
                                                on = ["nogl_id", "daydate"], 
                                                how="outer")

        # TODO : remove this line later       
        sceleton = sceleton.astype(str)
        sceleton["amazon_product_category_number"] = sceleton["amazon_product_category_number"].astype(str).str.split('.').str[0]
        sceleton["shopify_variant_barcode"] = sceleton["shopify_variant_barcode"].astype(str).str.split('.').str[0]
        sceleton["shopify_product_id"] = sceleton["shopify_product_id"].astype(str).str.split('.').str[0]
        sceleton["shopify_variant_id"] = sceleton["shopify_variant_id"].astype(str).str.split('.').str[0]
        sceleton["shopify_product_category_number"] = sceleton["shopify_product_category_number"].astype(str).str.split('.').str[0]
        

        unique_products = get_unique_products(sceleton)

        # TODO : remove this line later
        linkage_df = linkage_df.astype(str)
        linkage_df["shopify_variant_barcode"] = linkage_df["shopify_variant_barcode"].astype(str).str.split('.').str[0]
        linkage_df["shopify_variant_id"] = linkage_df["shopify_variant_id"].astype(str).str.split('.').str[0]
        linkage_df["variant_barcode"] = linkage_df["variant_barcode"].astype(str).str.split('.').str[0]
        linkage_df["variant_id"] = linkage_df["variant_id"].astype(str).str.split('.').str[0]

        logging.info("Data imported successfully")

    with log_time("Load transformation dataset"):
        #source_df = load_transformation_data("google_ads", prod_engine, dev_engine)
        source_df = load_transformation_data(transformation, engines["prod"], engines["dev"], channel_names)

        source_data_id_columns = {
            "google_ads": "segments.product_item_id",
            "klaviyo": "variant_sku",
            "google_analytics" : "ga_productsku",
            "amazon_sales" : "lineitem_seller_sku"
        }



        source_data_id_column = source_data_id_columns[transformation]
        mergable_products, IoU_scores = return_best_mergable_id(source_df=source_df, 
                                                             product_info=linkage_df, 
                                                             source_data_id_column=source_data_id_column,
                                                             sales_channel=channel_names)

                    
        best_mergable_id = max(IoU_scores, key=IoU_scores.get)
        print(IoU_scores)
        print("best mergable id for ", client_name, " is ", best_mergable_id)
        return best_mergable_id


if __name__ == "__main__":
    set_system_variables()
    #best_mergable_id(args.client_name, args.transformation)