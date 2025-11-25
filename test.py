#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import psycopg2
from datetime import datetime,date,timedelta
import os
import sys
from configAWSRDS import config
from sqlalchemy import create_engine
from Support_Functions import *

# test db engine
# Wefriends

import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument("--client_name", type=str, required=True)
# Parse the argument
args = parser.parse_args()

# get config params
section = args.client_name
params_wefriends = config(filename='databaseAWSRDS.ini', section=section)

#params_wefriends = config(filename='databaseAWSRDS.ini', section="wefriends")
#params_external = config(filename='databaseAWSRDS.ini', section="external")


engine_wefriends = create_engine('postgresql://'+params_wefriends.get("user")+":"+params_wefriends.get("password")+"@"+params_wefriends.get("host")+":5432/"+params_wefriends.get("database"),echo=False)
#engine_external = create_engine('postgresql://'+params_external.get("user")+":"+params_external.get("password")+"@"+params_external.get("host")+":5432/"+params_external.get("database"),echo=False)

print("engine wefriends db:", engine_wefriends)
#print("engine external db:", engine_external)

shopify_products_scd = import_data_AWSRDS(schema="shopify_old",table="products_scd",engine=engine_wefriends)
shopify_products_scd = shopify_products_scd[["_airbyte_products_hashid",
                                     "id", 
                                     "tags", 
                                     "title", 
                                     "handle", 
                                     "status", 
                                     "vendor", 
                                     "shop_url",
                                     "created_at", 
                                     "updated_at", 
                                     "product_type", 
                                     "published_at",
                                     "published_scope", 
                                     "template_suffix"]]
print(shopify_products_scd.head())

