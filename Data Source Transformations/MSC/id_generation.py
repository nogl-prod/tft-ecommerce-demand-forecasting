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

def check_if_table_exist(schema, table, engine):
    query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE  table_schema = '{schema}'
            AND    table_name   = '{table}'
        );
    """
    return pd.read_sql(query, engine).iloc[0,0]

def product_transform(client_name):

    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }

    products_dicts = {}
    with log_time("Importing data from AWS RDS"):
        
        for channel in channel_names:
            if channel == "shopify":
                table_name = channel+"_products"
                schema_name = "sc_" + channel
                channel_products = import_data_AWSRDS(schema=schema_name,table="product_info",engine=engines["prod"])
            elif channel == "amazon":
                table_name = channel+"_products"
                schema_name = "sc_" + channel
                channel_products = import_data_AWSRDS(schema="transformed",table=table_name,engine=engines["dev"])
            products_dicts[channel] = channel_products
    
    logging.info("Data imported from AWS RDS")

    with log_time("Cleaning data"):
        for i in products_dicts:
            products_dicts[i] = clean_data(products_dicts[i], channel=i)
    
    with log_time("Handling duplicates"):
        """
        where we have duplicates in one of the sales_channels we should merge the duplicates on variant_sku and variant_barcode
        * amazon SKU ABC-1 / Barcode 12345
        * amazon SKU ABC-2 / Barcode 12345
        * shopify SKU ABC-2 / Barcode 12345
        ------->>>>>
        * amazon SKU ABC-1 / shopify SKU "NaN" / Barcode 12345 / unique NOGL ID ss88s9
        * amazon SKU ABC-2 / shopify SKU ABC-2 / Barcode 12345 / unique NOGL ID 3sk3kk

        a) If a customer has duplicates with the same variant_barcode and product_status but 
        different variant_sku in one sales channel -> treat as individual variant, 
        generate individual forecasts and notify the customer about the duplicates

        b) If a customer has duplicates with the same variant_sku and product_status but
        different variant_barcode -> treat as individual variant, 
        generate individual forecasts and notify the customer about the duplicates
                
        """
        # For each group, check if there are multiple unique variant_sku

        consolidated_duplicates = {}
        for channel in products_dicts:
            logging.info("Handling duplicates in : ", channel)
            channel_duplicates = mark_duplicates(products_dicts[channel])
            channel_duplicates, channel_sku, channel_barcode = handle_channel_level_duplicates(channel_duplicates.copy())
            channel_duplicates = get_fused_variants(channel_duplicates, "local_nogl_id")
            channel_no_duplicates = products_dicts[channel][~products_dicts[channel]["variant_sku"].isin(channel_duplicates["variant_sku"])]
            channel_merged_variants = concat_duplicates_in_channel_level(channel_duplicates, channel_no_duplicates)
            consolidated_duplicates[channel] = channel_merged_variants[["local_nogl_id", "variant_sku", "variant_RRP", "variant_barcode", "product_status", "product_category"]]
        
        # concat all the channels in one dataframe 
        #concated_df = pd.concat(consolidated_duplicates.values())
        concated_df = pd.DataFrame()
        id_count = 0
        for channel in consolidated_duplicates:
            if id_count == 0:
                concated_df = consolidated_duplicates[channel]
                id_count = id_count + 1
            else:
                # concat the existing one
                concated_df = pd.concat([concated_df, consolidated_duplicates[channel]])
        
        
        concated_df = concated_df.reset_index(drop=True)
        concated_df = concated_df[["local_nogl_id", "variant_sku", "variant_barcode", "variant_RRP", "product_status", "product_category"]]
        # drop duplicates
        concated_df = concated_df.drop_duplicates(subset=["variant_sku", "variant_barcode"])

        merge_multi_variants = pd.DataFrame()
        id_count = 0
        for channel in consolidated_duplicates:

            channel_merged_variants = rename_columns_with_prefix(consolidated_duplicates[channel], prefix=channel+"_")
            logging.info("Merging channel : ", channel)
            if id_count == 0:
                merge_multi_variants = concated_df.merge(channel_merged_variants,
                    left_on = ["variant_sku", "variant_barcode"], 
                    right_on = [channel+ "_" + "variant_sku",  channel + "_" + "variant_barcode"], 
                    how="left")
                id_count += 1
            else:
                merge_multi_variants = merge_multi_variants.merge(channel_merged_variants,
                    left_on = ["variant_sku", "variant_barcode"], 
                    right_on = [channel+ "_" + "variant_sku",  channel + "_" + "variant_barcode"], 
                    how="left")

        # extract the rows that does not contains NaN values
        fused_variants = find_rows_without_nan(merge_multi_variants)
        no_fused_variants = merge_multi_variants[~merge_multi_variants.index.isin(fused_variants.index)]

        channel_no_fused_variants = {}
        for channel in channel_names:
            channel_no_fused_variants[channel] = no_fused_variants[no_fused_variants[channel + "_" + "variant_sku"].notna()]
            channel_no_fused_variants[channel] = channel_no_fused_variants[channel][["variant_barcode",
                                                                                     "variant_sku", 
                                                                                     channel+"_variant_sku", 
                                                                                     channel+"_variant_RRP",
                                                                                     channel+"_variant_barcode",
                                                                                     channel+"_product_status"
                                                                                     ]]
        merged_no_fused_barcode = pd.DataFrame()
        merged_no_fused_sku = pd.DataFrame()
        channel_count = 0
        for channel in channel_no_fused_variants:
            if channel_count == 0:
                merged_no_fused_barcode = channel_no_fused_variants[channel]
                merged_no_fused_sku = channel_no_fused_variants[channel]
                channel_count += 1
            else:
                merged_no_fused_barcode = merged_no_fused_barcode.merge(channel_no_fused_variants[channel],
                    left_on = "variant_barcode", 
                    right_on = "variant_barcode", 
                    how="left")
                merged_no_fused_sku = merged_no_fused_sku.merge(channel_no_fused_variants[channel],
                    left_on = "variant_sku", 
                    right_on = "variant_sku", 
                    how="left")
                
            
        fused_on_sku = find_rows_without_nan(merged_no_fused_sku)
        fused_on_barcode = find_rows_without_nan(merged_no_fused_barcode)
        fused_on_both = find_rows_without_nan(merge_multi_variants)

        columns_to_keep = []
        for channel in channel_names:
            columns_to_keep.append(channel + "_variant_sku")
            columns_to_keep.append(channel + "_variant_RRP")
            columns_to_keep.append(channel + "_variant_barcode")
            columns_to_keep.append(channel + "_product_status")
        
        columns_to_keep_sku = columns_to_keep.copy()
        columns_to_keep_sku.append("variant_sku")
        fused_on_sku = fused_on_sku[columns_to_keep_sku]

        columns_to_keep_barcode = columns_to_keep.copy()
        columns_to_keep_barcode.append("variant_barcode")
        fused_on_barcode = fused_on_barcode[columns_to_keep_barcode]

        columns_to_keep_both = columns_to_keep.copy()
        columns_to_keep_both.append("variant_sku")
        columns_to_keep_both.append("variant_barcode")
        fused_on_both = fused_on_both[columns_to_keep_both]

        # drop local_id from merge_multi_variants
        merge_multi_variants = merge_multi_variants.drop(columns=["local_nogl_id"])
        no_fused_variants = no_fused_variants.drop(columns=["local_nogl_id"])
        # drop the rows that contains variant_barcode of fused_on_barcode from no_fused_variants
        no_fused_variants = no_fused_variants[~no_fused_variants["variant_barcode"].isin(fused_on_barcode["variant_barcode"])]
        # drop the rows that contains variant_sku of fused_on_sku from no_fused_variants
        no_fused_variants = no_fused_variants[~no_fused_variants["variant_sku"].isin(fused_on_sku["variant_sku"])]
        no_fused_variants_cleaned = no_fused_variants[columns_to_keep]

        # drop the rows that contains variant_barcode of fused_on_barcode from merge_multi_variants
        fused_on_both_cleaned = fused_on_both[columns_to_keep]
        fused_on_sku_cleaned = fused_on_sku[columns_to_keep]
        fused_on_barcode_cleaned = fused_on_barcode[columns_to_keep]

        fused_variants_cleaned = fused_variants[columns_to_keep]
        
        # concat fused_on_sku, fused_on_barcode, fused_on_both, no_fused_variants_cleaned
        final_products = pd.concat([fused_on_sku_cleaned, fused_on_barcode_cleaned, fused_on_both_cleaned, no_fused_variants_cleaned], axis=0)
        final_products = final_products.reset_index(drop=True)

        try:
            # check if there are still duplicates
            if len(final_products[final_products.duplicated(subset=columns_to_keep, keep=False)]) > 0:
                logging.info("There are still duplicates!!!")
                logging.info("Duplicates: ", final_products[final_products.duplicated(subset=columns_to_keep, keep=False)])
                logging.info("Dropping the duplicates...")
                final_products = final_products.drop_duplicates(subset=columns_to_keep)
        except:
            logging.info("No duplicates found at final merge level")
        

        logging.info("Duplicates handled")

    with log_time("Unique id generation"):
        # add unique id to final_products
        final_products["nogl_id"] = final_products.apply(hash_row, axis=1)
        logging.info("Unique id generated")

    with log_time("Enrichment of channel wise product tables"):
        for channel in products_dicts:
            try:
                if channel == "shopify":
                    new_field = "variant_id"
                    columns_to_keep.append(channel+"_variant_id")
                elif channel == "amazon":
                    new_field = "variant_asin"
                    columns_to_keep.append(channel+"_variant_asin")
            except:
                logging.info("Channel is not found, codebase should be updated manually")
            

            print("here 1")
            channel_final_products_enriched = add_field(products_dicts[channel], final_products, channel=channel, new_field=new_field)
            final_products = final_products.merge(channel_final_products_enriched, 
                left_on=[channel+"_variant_sku", channel+"_variant_barcode", channel+"_product_status"], 
                right_on=["variant_sku", "variant_barcode", "product_status"], 
                how="left")
        

        print("here 2")
        columns_to_keep.append("nogl_id")
        final_products = final_products[columns_to_keep]
        # TODO : Make it dynamic later on
        if len(channel_names)>1:
            final_products["variant_sku"] = final_products.apply(lambda x: get_value_or_fill(x["amazon_variant_sku"], x["shopify_variant_sku"]), axis=1)
            final_products["variant_barcode"] = final_products.apply(lambda x: get_value_or_fill(x["amazon_variant_barcode"], x["shopify_variant_barcode"]), axis=1)
            final_products["variant_id"] = final_products["shopify_variant_id"]
            final_products["variant_asin"] = final_products["amazon_variant_asin"]
        else:
            if channel_names[0] == "amazon":
                final_products["variant_sku"] = final_products["amazon_variant_sku"]
                final_products["variant_barcode"] = final_products["amazon_variant_barcode"]
                # assign variant_id to none
                final_products["variant_id"] = None
                final_products["variant_asin"] = final_products["amazon_variant_asin"]
            else:
                final_products["variant_sku"] = final_products["shopify_variant_sku"]
                final_products["variant_barcode"] = final_products["shopify_variant_barcode"]
                final_products["variant_id"] = final_products["shopify_variant_id"]
                # assign variant_asin to none
                final_products["variant_asin"] = None

            
        print("here 3")

        # drop duplicates in nogl id colummn before saving it
        final_products = final_products.drop_duplicates(subset=["nogl_id"])
        # reset index
        final_products = final_products.reset_index(drop=True)
        # convert all "nan" values into None 
        final_products = final_products.replace("nan", None)
        final_products = final_products.replace(np.nan, None)
                        

    with log_time("Exporting data to AWS RDS"):

            #final_products.to_sql('product_info_test', con=engines["dev"], schema="product", if_exists='replace', index=False) # drops old table and creates new empty table
            final_products = final_products.drop(columns=["variant_id"])
            final_products = final_products.rename(columns={"nogl_id": "variant_id"})

            if check_if_table_exist(schema="product", table="product_info", engine=engines["dev"]):
                # if exist then add new rows to db
                logging.info("product_info table is exist in db")
                # get the existing rows from db
                existing_df = import_data_AWSRDS(schema="product",table="product_info",engine=engines["dev"])
                # find similar rows between existing_rows and final_products (ignore the variant_id)
                cols = list(existing_df.columns)
                # remove "variant_id"
                cols.remove("variant_id")
                # try to merge final_products with existing_rows
                merged_df = final_products[cols].merge(existing_df,
                    left_on=cols,
                    right_on=cols,
                    how="right")
                
                # find nan values in variant_id column
                new_rows = merged_df[merged_df["variant_id"].isna()]
                # 
                new_products = new_rows[cols].merge(final_products,
                    left_on=cols,
                    right_on=cols,
                    how="right")
                
                # concat new products to existing_df
                existing_df = pd.concat([existing_df, new_products])
                # drop duplicates
                existing_df = existing_df.drop_duplicates(subset=["variant_id"])
                # reset index
                existing_df = existing_df.reset_index(drop=True)
                
                final_products = existing_df
            
            print(final_products[["variant_id"]].head(10))
            # Import Tables
            for channel in channel_names:
                if channel == "amazon":
                    table_name = channel+"_products"
                    schema_name = "sc_" + channel
                    amazon_products = import_data_AWSRDS(schema="transformed",table=table_name,engine=engines["dev"])
                    merge_amazon_products = final_products[['variant_id', 'amazon_variant_sku', 'amazon_variant_barcode', 'amazon_product_status', 'amazon_variant_asin']]
                    merge_amazon_products = merge_amazon_products.rename(columns={col:col.replace("amazon_", "") for col in merge_amazon_products.columns})
                    amazon_products["variant_barcode"] = amazon_products["variant_barcode"].replace("None", np.nan)
                    amazon_products["variant_barcode"] = amazon_products["variant_barcode"].fillna(amazon_products["product_id"])
                    final_amazon_products = amazon_products.merge(merge_amazon_products,
                        left_on=["variant_asin", "variant_sku", "variant_barcode", "product_status"],
                        right_on=["variant_asin", "variant_sku", "variant_barcode", "product_status"],
                        how="left")
                    
                    final_amazon_products = product_status_mapping(final_amazon_products, channel=None)
                    print(final_amazon_products[["variant_id"]].head(10))
                    print(final_amazon_products["variant_id"].unique())
                    # drop values if vaariant_id is non
                    #final_amazon_products = final_amazon_products[final_amazon_products["variant_id"].notna()]
                    final_amazon_products.to_sql('product_info', con=engines["dev"], schema="sc_amazon", if_exists='replace', index=False) # drops old table and creates new empty table

                elif channel == "shopify":
                    table_name = channel+"_products"
                    schema_name = "sc_" + channel
                    shopify_products = import_data_AWSRDS(schema=schema_name,table="product_info",engine=engines["prod"])
                    shopify_products = shopify_products.rename(columns={"variant_id": "shopify_variant_id"})
                    merge_shopify_products = final_products[['variant_id', 'shopify_variant_sku', 'shopify_variant_barcode', 'shopify_product_status', 'shopify_variant_id']]
                    # remove channel name from column names
                    merge_shopify_products = merge_shopify_products.rename(columns={"shopify_variant_id":"channel_variant_id"})
                    merge_shopify_products = merge_shopify_products.rename(columns={col:col.replace("shopify_", "") for col in merge_shopify_products.columns})
                    merge_shopify_products = merge_shopify_products.rename(columns={"channel_variant_id":"shopify_variant_id"})
                    # merge shopify_products with final_products to add nogl_id
                    # fill_na in variant_barcode with variant_id in shopify_products
                    # replace None values of variant barcode with variant_id since assumption is hold.for shopify
                    shopify_products["variant_barcode"] = shopify_products["variant_barcode"].replace("", np.nan)
                    shopify_products["variant_barcode"] = shopify_products["variant_barcode"].fillna(shopify_products["shopify_variant_id"])
                    
                    final_shopify_products = shopify_products.merge(merge_shopify_products,
                        left_on=["shopify_variant_id", "variant_sku", "variant_barcode", "product_status"],
                        right_on=["shopify_variant_id", "variant_sku", "variant_barcode","product_status"],
                        how="left")
                    #final_shopify_products = final_shopify_products[final_shopify_products["variant_id"].notna()]
                    final_shopify_products.to_sql('product_info', con=engines["dev"], schema="sc_shopify", if_exists='replace', index=False) # drops old table and creates new empty table
                else:
                    logging.info("Channel is not found, codebase should be updated manually")

            
            if len(channel_names)>1:
                final_products = product_status_mapping(final_products, channel="amazon")
            
            # first check if product_info is exist in db or not. İf not save directly to db if yes then just add different rows
            # check if product_info is exist in db or not




                
                

            final_products.to_sql('product_info', con=engines["dev"], schema="product", if_exists='replace', index=False) # drops old table and creates new empty table
            logging.info("Data exported to AWS RDS")


if __name__ == "__main__":
    set_system_variables()
    product_transform(args.client_name)
