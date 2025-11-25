# Imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import time
from google.oauth2 import service_account
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

def facebook_ads(client_name):
    channel_names = get_channel_name(client_name)
    engines = {
        "dev" : get_db_connection(dbtype="dev", client_name=client_name),
        "prod" : get_db_connection(dbtype="prod", client_name=client_name)
    }
    productxday_skeleton = import_data_AWSRDS(schema="product",table="productsxdays_sceleton",engine=engines["dev"])
    insights_df = load_insights(section=client_name)
    ads_df = load_ads()
    adsets_df = load_adsets()
    campaigns_df = load_campaigns()

    # rename table features
    df_dic = {"ads":ads_df, "adsets":adsets_df, "campaigns":campaigns_df}

    for df in df_dic:
        for c in list(df_dic.get(df).columns):
            df_dic.get(df).rename(columns={c:df+"_"+c}, inplace=True)

    campaigns_df_current = campaigns_df.sort_values(by=["campaigns_id", "campaigns_loaded_at", "campaigns_received_at"], ascending=[True, False, False])
    campaigns_df_current = campaigns_df_current.groupby("campaigns_id", as_index=False).first()

    adsets_df_current = adsets_df.sort_values(by=["adsets_id", "adsets_loaded_at", "adsets_received_at"], ascending=[True, False, False])
    adsets_df_current = adsets_df_current.groupby("adsets_id", as_index=False).first()


    ads_df_current = ads_df.sort_values(by=["ads_id", "ads_loaded_at", "ads_received_at"], ascending=[True, False, False])
    ads_df_current = ads_df_current.groupby("ads_id", as_index=False).first()


    # generate id_mapping dictionary for ads, ad sets and campaigns
    id_mapping_current = adsets_df_current.merge(ads_df_current, how="left", left_on="adsets_id", right_on="ads_adset_id").drop_duplicates()
    id_mapping_current = id_mapping_current.merge(campaigns_df_current, how="left", left_on="adsets_campaign_id", right_on="campaigns_id").drop_duplicates()
    id_mapping_current = id_mapping_current[["campaigns_account_id",
                                            "campaigns_id",
                                            "campaigns_name",
                                            "campaigns_effective_status",
                                            "adsets_id",
                                            "adsets_name",
                                            "adsets_effective_status",
                                            "adsets_daily_budget",
                                            "ads_id",
                                            "ads_name",
                                            "ads_status",
                                            "ads_bid_amount",
                                            "ads_bid_type"
                                            ]]
    unique_product_categories = []
    for channel in channel_names:
        for i in list(productxday_skeleton[channel+"_product_category"].unique()):
            unique_product_categories.append(i)


    # add special names
    to_append = ["Mug","Shirt"]
    for a in to_append:
        unique_product_categories.append(a)
        
    unique_product_categories.append("Rest")

    id_mapping_current["extract_campaign"] = id_mapping_current.campaigns_name.astype(str)
    id_mapping_current["extract_adset"] = id_mapping_current.adsets_name.astype(str)
    id_mapping_current["extract_ad"] = id_mapping_current.ads_name.astype(str)

    for p in unique_product_categories:
        id_mapping_current["extract_campaign"] = id_mapping_current["extract_campaign"].apply(lambda x: str(p) if (str(p) in x) else x)
        id_mapping_current["extract_adset"] = id_mapping_current["extract_adset"].apply(lambda x: str(p) if (str(p) in x) else x)
        id_mapping_current["extract_ad"] = id_mapping_current["extract_ad"].apply(lambda x: str(p) if (str(p) in x) else x)
        
        id_mapping_current["extract_campaign"] = id_mapping_current["extract_campaign"].apply(lambda x: x if (x in unique_product_categories) else "")
        id_mapping_current["extract_adset"] = id_mapping_current["extract_adset"].apply(lambda x: x if (x in unique_product_categories) else "")
        id_mapping_current["extract_ad"] = id_mapping_current["extract_ad"].apply(lambda x: x if (x in unique_product_categories) else "")



    
    id_mapping_current["mapping"] = id_mapping_current.apply(lambda x: pick_extract(x.extract_campaign, x.extract_adset, x.extract_ad), axis=1)

    id_mapping_current.drop(columns=["extract_campaign", "extract_adset", "extract_ad"], inplace=True)

    id_mapping_current["mapping"] = id_mapping_current["mapping"].str.replace('Shirt','T-Shirt')
    id_mapping_current["mapping"] = id_mapping_current["mapping"].str.replace('Mug','Tasse')

    spends = insights_df.merge(id_mapping_current, how="left", left_on="ad_id", right_on="ads_id")

    fb_daily_data = preprocess_facebook_data(insights_df)
    complete_facebook_data = facebook_merge_into_product_x_day_skeleton(fb_daily_data, productxday_skeleton)

    t = Timer("Upload")
    complete_facebook_data.rename(columns={"nogl_id":"variant_id"}, inplace=True)
    complete_facebook_data.to_sql("msc_facebook_ads", con = engines["dev"], schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
    #spends.to_sql("facebook_ad_spends", con = engines["dev"], schema="demand_shaping", if_exists='replace', index=False, chunksize=1000, method="multi")
    t.end()


if __name__ == "__main__":
    facebook_ads(args.client_name)