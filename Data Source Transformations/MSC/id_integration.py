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

def id_integration(client_name):

    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }

    products_dicts = {}
    with log_time("Importing data from AWS RDS"):
        
        schema_name = "product"
        linkage_df = import_data_AWSRDS(schema=schema_name,table="product_info",engine=engines["dev"])
        # TODO : update this later on
        # rename variant_id to nogl_id
        linkage_df.rename(columns={"variant_id": "nogl_id"}, inplace=True)
        # create a new column with shopify_variant_id and name variant_id
        linkage_df["variant_id"] = linkage_df["shopify_variant_id"]


        sceleton_dict = {}
        for channel_name in channel_names:
            # TODO : change transformed schema name later to sc_channel name
            if channel_name == "shopify":
                sceleton_dict[channel_name] = import_data_AWSRDS(schema="sc_shopify", table=channel_name+"_productsxdays_sceleton", engine=engines["prod"])
            elif channel_name == "amazon":
                sceleton_dict[channel_name] = import_data_AWSRDS(schema="transformed", table=channel_name+"_productsxdays_sceleton", engine=engines["dev"])
                sceleton_dict[channel_name]["variant_barcode"] = sceleton_dict[channel_name]["product_id"]
            else:
                raise ValueError("Channel name is not valid")
            linkage_df[channel_name+"_variant_sku"] = linkage_df[channel_name+"_variant_sku"].astype(str).str.split('.').str[0]
            linkage_df[channel_name+"_variant_barcode"] = linkage_df[channel_name+"_variant_barcode"].astype(str).str.split('.').str[0]

        # convert shopify_variant_id to string
        # TODO : variant_id should be changed to shopify_variant_id later and nogl id will be renamed as variant_id
        linkage_df["shopify_variant_id"] = linkage_df["shopify_variant_id"].astype(str).str.split('.').str[0]
        linkage_df["variant_id"] = linkage_df["variant_id"].astype(str).str.split('.').str[0]
        linkage_df["variant_barcode"] = linkage_df["variant_barcode"].astype(str).str.split('.').str[0]
        linkage_df["variant_sku"] = linkage_df["variant_sku"].astype(str).str.split('.').str[0]


        logging.info("Data imported successfully")

    with log_time("Enrich sceleton with unique variant id (nogl_id).."):

        
        id_df_dict = {}
        for channel in channel_names:
            id_df_channel = linkage_df.copy()
            # Identify columns with the prefix 'shopify'
            channel_columns = [col for col in id_df_channel.columns if col.startswith(channel)]
            # Filter rows where any of the shopify columns is non-NA
            id_df_channel = id_df_channel[(id_df_channel[channel_columns] != 'None').all(axis=1)]
            channel_columns.append("nogl_id")
            id_df_channel = id_df_channel[channel_columns]
            id_df_dict[channel] = id_df_channel

        refactored_sceleton_dict = {}
        for channel in channel_names:
            channel_sceleton = add_nogl_id_to_sceleton(id_df_dict[channel], sceleton_dict[channel], channel)
            refactored_sceleton_dict[channel] = channel_sceleton

        logging.info("Sceleton enriched successfully for channel wise")
        
    with log_time("Generate global product sceleton.."):

        custom_sceleton_cols = [
            "nogl_id",
            "daydate"
            "product_id",
            "variant_sku",
            "variant_barcode",
            "shopify_variant_id",
            "amazon_variant_asin",
            "product_category_number",
            "product_category",
            "product_status"
        ]

        custom_sceleton_dict = {}
        for channel in channel_names:
            if channel == "shopify":
                custom_sceleton_dict[channel] = refactored_sceleton_dict[channel][[
                    "nogl_id",
                    "daydate",
                    "product_id",
                    "variant_sku",
                    "variant_barcode",
                    "variant_id",
                    "variant_RRP",
                    "product_category_number",
                    "product_category",
                    "product_status"
                ]]

            elif channel == "amazon":
                custom_sceleton_dict[channel] = refactored_sceleton_dict[channel][[
                    "nogl_id",
                    "daydate",
                    "product_id",
                    "variant_sku",
                    "variant_barcode",
                    "variant_asin",
                    "variant_RRP",
                    "product_category_number",
                    "product_category",
                    "product_status"
                ]]

            else:
                raise ValueError("Channel name is not valid")

            custom_sceleton_dict[channel] = custom_sceleton_dict[channel].add_prefix(channel+"_")
            custom_sceleton_dict[channel].rename(columns={channel+'_nogl_id': 'nogl_id'}, inplace=True)
            custom_sceleton_dict[channel].rename(columns={channel+'_daydate': 'daydate'}, inplace=True)

        logging.info("Merging starting..")

        custom_sceleton = pd.DataFrame()
        id_count = 0
        for channel in channel_names:  
            if id_count == 0:
                custom_sceleton = custom_sceleton_dict[channel]
                id_count += 1
            else:
                custom_sceleton = custom_sceleton.merge(custom_sceleton_dict[channel], on=["nogl_id", "daydate"], how="outer")
                id_count += 1
        
        # convert all columns to string
        custom_sceleton[custom_sceleton.columns] = custom_sceleton[custom_sceleton.columns].astype(str)
        # convert proudct_id_shopify to int string froamt
        custom_sceleton["shopify_product_id"] = custom_sceleton["shopify_product_id"].astype(str).str.split('.').str[0]
        custom_sceleton["shopify_product_category_number"] = custom_sceleton["shopify_product_category_number"].astype(str).str.split('.').str[0]
        custom_sceleton["amazon_product_category_number"] = custom_sceleton["amazon_product_category_number"].astype(str).str.split('.').str[0]

        custom_sceleton = custom_sceleton.replace("nan", None)
        custom_sceleton = custom_sceleton.replace("None", None)

        logging.info("Merging completed successfully")

    with log_time("Exporting data to AWS RDS"):
        for channel in channel_names:
            custom_sceleton_dict[channel].to_sql(channel+'_productsxdays_sceleton', con = engines["dev"], schema="sc_"+channel, if_exists='replace',index=False) #drops old table and creates new empty table

        custom_sceleton.to_sql('productsxdays_sceleton', con = engines["dev"], schema="product", if_exists='replace',index=False) #drops old table and creates new empty table   




if __name__ == "__main__":
    set_system_variables()
    id_integration(args.client_name)
