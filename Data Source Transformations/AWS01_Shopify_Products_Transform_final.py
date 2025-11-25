#!/usr/bin/env python
# coding: utf-8

# # TO DO

# - Dynamic CDC needed? -> Does Shopify always store non-active "old" products?
# - What to do with sets made from existing skus?
#     - Currently you have to put in new variants and product_ids in Shopify
#     - Currently it will confirm quantity just 1 for the new sku-id
# - Check how to handle active vs. non-active products (use non-active for training anyways) -> TS nur so lange wie Produkt aktiv war / Daten vorhanden sind —> kann ich so trainieren? --> Create column with product_active (and fill all unactive dates, later remove those for training?)
# - feature naming translation in the beginning, so code is resitant to raw data feature name changes

# # 0. Installs

# # 1. Imports/Options

# ## 1.1 External imports

# In[1]:


import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, date, timedelta


# ## 1.2 Internal imports

# In[2]:


import sys
import os
prefix = '/opt/ml'
src  = os.path.join(prefix, 'processing/input/')
sys.path.append(src)


# import the 'config' funtion from the config.py file

from configAWSRDS import config

# from support functions:

from Support_Functions import *
from static_variables import datacollection_startdate


# In[3]:

import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add argument
parser.add_argument("--client_name", type=str, required=True)
# Parse the argument
args = parser.parse_args()


# get config params
section = args.client_name
print("client name:", section)
filename = '/opt/ml/processing/input/databaseAWSRDS.ini'
params = config(section=section,filename=filename)

# create sqlalchemy connection
from sqlalchemy import create_engine    

engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)


# ## 1.3 Options

# In[5]:


pd.set_option('display.max_columns', None)


# # 2. Load data

# ## 2.1 Connect to DB

# ## 2.2 Load data in pandas dataframe

# In[6]:


shopify_products_scd = import_data_AWSRDS(schema="shopify",table="products_scd",engine=engine)
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


# In[7]:


shopify_products_variants = import_data_AWSRDS(schema="shopify",table="products_variants",engine=engine)
shopify_products_variants = shopify_products_variants[["_airbyte_products_hashid", 
                                               "id", 
                                               "sku",
                                               "grams",
                                               "price",
                                               "title",
                                               "weight",
                                               "barcode",
                                               "option1",
                                               "option2",
                                               "option3",
                                               "taxable", 
                                               "image_id", 
                                               "position",
                                               "tax_code",
                                               "created_at",
                                               "updated_at",
                                               "weight_unit",
                                               "compare_at_price",
                                               "inventory_policy", 
                                               "inventory_item_id",
                                               "requires_shipping",
                                               "inventory_quantity",
                                               "presentment_prices",
                                               "fulfillment_service",
                                               "inventory_management",
                                               "old_inventory_quantity"]]


# # 3. Transform & clean data

# In[8]:


# prepare variants dataframe for merging
rename_dfcolumns(shopify_products_variants,"variant_") # rename each column to track source after merging
shopify_products_variants.drop_duplicates(inplace=True) # remove duplicates


# In[9]:


# prepare products_scd dataframe for merging
rename_dfcolumns(shopify_products_scd,"product_") # rename each column to track source after merging


# In[10]:


# merge both data frames
shopify_products = shopify_products_variants.merge(shopify_products_scd, how="left", on="_airbyte_products_hashid")


# In[11]:


# clean results, mostly removing duplicates (old updates)
shopify_products["product_updated_at"] = pd.to_datetime(shopify_products["product_updated_at"], utc=True) # convert to datetime
shopify_products["variant_updated_at"] = pd.to_datetime(shopify_products["variant_updated_at"], utc=True) # convert to datetime
shopify_products.drop_duplicates(inplace=True) # remove duplicates
shopify_products["max_updated_at"] = shopify_products[["variant_updated_at","product_updated_at"]].values.max(1) # get max timestamp from updated
shopify_products = shopify_products.loc[shopify_products["variant_sku"] != ""] # remove "" sku values

shopify_products.sort_values(["variant_sku","max_updated_at","variant_inventory_quantity"],axis=0 ,ascending=[True,False,True],inplace=True) # sort to remove duplicates, always keeping most current row based on last update and lowest inventory_quantity (assumption)
shopify_products = shopify_products.groupby("variant_sku").first() # keep first row per sku group after sorting, only leaving most current
shopify_products.reset_index(inplace=True) # reset index after grouping
shopify_products = shopify_products.astype({"product_id":'int64'}) # change from float to int64, as it is an id number
shopify_products["variant_sku"] = shopify_products["variant_sku"].apply(lambda x: x.partition(',')[0]) #clean variant_sku from weird names such as "160274744,_1_1_1"


# # 4. Finalization

# ## 4.1 Finalize shopify_products

# In[12]:

# dict with dtypes for transformation
dtypes_dict = {"variant_sku":"str",
"_airbyte_products_hashid":"str",
"variant_id":"str",
"variant_grams":"float64",
"variant_price":"float64",
"variant_title":"str",
"variant_weight":"float64",
"variant_barcode":"str", #changed to str
"variant_option1":"str",
"variant_option2":"str",
"variant_option3":"str",
"variant_taxable":"boolean",
"variant_image_id":"int64",
"variant_position":"int64",
"variant_tax_code":"str", # drop
"variant_created_at":"datetime",
"variant_updated_at":"datetime",
"variant_weight_unit":"str",
"variant_compare_at_price":"float64",
"variant_inventory_policy":"str", # later boolean: continue = True, deny = False
"variant_inventory_item_id":"int64",
"variant_requires_shipping":"boolean",
"variant_inventory_quantity":"int64",
"variant_presentment_prices":"str", # drop
"variant_fulfillment_service":"str",
"variant_inventory_management":"str", # later boolean: somestring = True, else = False 
"variant_old_inventory_quantity":"int64", 
"product_id":"int64",
"product_tags":"str",
"product_title":"str",
"product_handle":"str",
"product_status":"str",
"product_vendor":"str",
"product_shop_url":"str",
"product_created_at":"datetime",
"product_updated_at":"datetime",
"product_product_type":"category",
"product_published_at":"datetime",
"product_published_scope":"str",
"product_template_suffix":"str",
"max_updated_at":"datetime"}


# ### 4.1.1 Fill NaN values

# In[13]:


# create table to allocate fillna value for each datatype

# function allocating fillna_values to each dtype
def find_fillna_value(dtype):
    if (dtype == "int64") or (dtype == "float64"):
        return 0
    if (dtype == "str"):
        return ""
    if (dtype == "boolean"):
        return False
    if (dtype == "datetime"):
        return datacollection_startdate
    if (dtype == "category"):
        return ""
    
# convert dict to dataframe
transformation_df = pd.DataFrame.from_dict(dtypes_dict, orient="index").reset_index().rename(columns={0:"dtype","index":"feature"})

# add fillna_value according to function
transformation_df["fillna_value"] = transformation_df["dtype"].apply(lambda x: find_fillna_value(x))

# add number of nan values to each feature
transformation_df = transformation_df.merge(pd.DataFrame(shopify_products.isna().sum()).reset_index().rename(columns={0:"number_of_NaN_values","index":"feature"}), how="left", on="feature")


# #### 4.1.1.1 Special check and fill NaN values in datetime columns

# In[14]:


# create list of columns for checking and replacing fillna values with datetime format
list_datetime_columns_to_check = list(transformation_df.loc[(transformation_df["dtype"] == "datetime") & (transformation_df["number_of_NaN_values"] > 0)].feature)

# check first if product_created_at has NaN values and if yes, fill with datacollection_startdate
if "product_created_at" in list_datetime_columns_to_check:
    shopify_products["product_created_at"].fillna(datacollection_startdate, inplace=True)

# individual fillna cleaning
shopify_products['variant_created_at'].fillna(shopify_products['product_created_at'], inplace=True)
shopify_products['variant_updated_at'].fillna(shopify_products['product_updated_at'], inplace=True)
shopify_products['product_updated_at'].fillna(shopify_products['product_created_at'], inplace=True)
shopify_products['product_published_at'].fillna(shopify_products['product_created_at'], inplace=True)
shopify_products['max_updated_at'].fillna(shopify_products['product_created_at'], inplace=True)


# In[15]:


# convert datetime column to datetime with utc = True
list_datetime_columns= list(transformation_df.loc[(transformation_df["dtype"] == "datetime")].feature)

for c in list_datetime_columns:
    shopify_products[c] = pd.to_datetime(shopify_products[c], utc=True)


# #### 4.1.1.2 Special check and fill NaN values in numeric (int and float) columns

# In[16]:


# individual fillna cleaning
shopify_products["variant_compare_at_price"].fillna(shopify_products["variant_price"], inplace=True)


# In[17]:


# convert str "" to 0 for all numeric
list_numeric_columns_to_replace = list(transformation_df.loc[(transformation_df["dtype"] == "int64") | (transformation_df["dtype"] == "float64")].feature)
for c in list_numeric_columns_to_replace:
    shopify_products[c].replace("",'0', inplace=True)


# #### 4.1.1.3 Special check and fill NaN values in str columns

# In[18]:


# not necessary as of now


# #### 4.1.1.4 Special check and fill NaN values in boolean columns

# In[19]:


# not necessary as of now


# #### 4.1.1.5 Special check and fill NaN values in category columns

# In[20]:


# not necessary as of now
shopify_products["product_product_type"] = shopify_products["product_product_type"].replace("","None")

# remove special characters
columns_for_special_character_removal = ["variant_sku",
                                         "variant_title",
                                         "product_title",
                                         "product_product_type", 
                                         "variant_option1",
                                         "variant_option2",
                                         "variant_option3", 
                                         "product_tags", 
                                         "product_handle"]
for k in dtypes_dict.keys():
    if (dtypes_dict.get(k) in ["category", "str"]) and (k in columns_for_special_character_removal):
        print("Replacing special characters for", k)
        shopify_products[k] = shopify_products[k].astype(dtypes_dict.get(k))
        shopify_products[k] = shopify_products[k].str.replace(r'[^\w\s]', '', regex=True)

# #### 4.1.1.6 Final check and fill NaN values for rest of columns still having NaN values

# In[21]:


# update NaN value count in transformation_df
transformation_df.drop(columns=["number_of_NaN_values"],inplace=True)
transformation_df = transformation_df.merge(pd.DataFrame(shopify_products.isna().sum()).reset_index().rename(columns={0:"number_of_NaN_values","index":"feature"}), how="left", on="feature")


# In[22]:


# Create dict with columns still having NaN values
dict_restNaNstofill = pd.Series(transformation_df.loc[(transformation_df["number_of_NaN_values"] > 0)].fillna_value.values,index=transformation_df.loc[(transformation_df["number_of_NaN_values"] > 0)].feature).to_dict()
# Fillna rest using dict
shopify_products.fillna(value=dict_restNaNstofill, inplace=True)


# In[23]:


# update NaN value count in transformation_df
transformation_df.drop(columns=["number_of_NaN_values"],inplace=True)
transformation_df = transformation_df.merge(pd.DataFrame(shopify_products.isna().sum()).reset_index().rename(columns={0:"number_of_NaN_values","index":"feature"}), how="left", on="feature")


# In[24]:


# check if still NaN values existing, and raise an error if the case
if len(list(transformation_df.loc[(transformation_df["number_of_NaN_values"] > 0)].feature)) > 0:
    raise NameError("Still NaN values existing.")


# ### 4.1.2 Change dtypes

# In[25]:


# convert all dtypes except datetime, as this was done previously with pd.to_datetime
dict_dtypechanges = pd.Series(transformation_df.loc[(transformation_df["dtype"] != "datetime")].dtype.values,index=transformation_df.loc[(transformation_df["dtype"] != "datetime")].feature).to_dict()
#shopify_products = shopify_products.astype(dict_dtypechanges)


# Iterate over the columns and attempt to cast each one individually
for col, dtype in dict_dtypechanges.items():
    try:
        shopify_products[col] = shopify_products[col].astype(dtype)
    except ValueError as e:
        print(f"Error converting column {col} to {dtype}: {e}")


# ### 4.1.3 Feature engineering / transformation

# In[26]:


# variant_inventory_policy boolean transformation --> continue = True, deny = False
shopify_products["variant_inventory_policy"] = shopify_products["variant_inventory_policy"].replace(["continue","deny"],[True, False]).astype("boolean")

# variant_inventory_management boolean transformation --> shopify = True, "" = False
shopify_products["variant_inventory_management"] = shopify_products["variant_inventory_management"].apply(lambda x: True if x != "" else False).astype("boolean")

# give categories numbering product_product_type -> later renamed to product_category
shopify_products["product_category_number"] = shopify_products["product_product_type"].cat.codes


# ### 4.1.4 Column renaming

# In[27]:


# rename price to RRP (recommended retail price aka pricebeforediscount)
shopify_products.rename(columns={"product_product_type":"product_category","variant_price":"variant_RRP","variant_inventory_policy":"variant_inventory_policy_continue", "variant_inventory_management":"variant_inventory_management_used", "max_updated_at":"variant_max_updated_at"}, inplace=True)

# set negative variant_inventory_quantity to 0
shopify_products.loc[shopify_products["variant_inventory_quantity"] < 0, "variant_inventory_quantity"] = 0

# In[28]:


#shopify_products


# ## 4.2 Create dataframe sceleton (sku x day) for merging and other data pre-processing

# In[29]:


shopify_products_sceleton = shopify_products[["variant_sku",
                                    "variant_id",
                                    "variant_grams",
                                    "variant_RRP",
                                    "variant_taxable",
                                    "variant_position",
                                    "variant_created_at",
                                    "variant_inventory_item_id",
                                    "variant_requires_shipping",
                                    "variant_inventory_management_used",
                                    "product_id",
                                    "product_status",
                                    "product_category",
                                    "product_category_number",
                                    "product_published_at",
                                    "product_published_scope",
                                    "variant_max_updated_at"]]


# In[30]:


dateframe = create_dateframe()
print("maximum date")
print("Dateframe: ", dateframe.daydate.max())
print("Dateframe maximum date is: ", dateframe.daydate.max())


# In[31]:


# perform merge
shopify_products_sceleton["key"] = 1
dateframe["key"] = 1
shopify_products_sceleton = dateframe.copy().merge(shopify_products_sceleton, how="left", on="key").drop(columns=["key"])


print("Sceleton maximum date is after merging with dateframe: ", shopify_products_sceleton.daydate.max())

# In[32]:


shopify_products_sceleton = shopify_products_sceleton[["product_category_number",
                                    "product_category",
                                    "product_id",
                                    "variant_sku",
                                    "variant_id",
                                    "daydate",
                                    "variant_grams",
                                    "variant_RRP",
                                    "variant_taxable",
                                    "variant_position",
                                    "variant_created_at",
                                    "variant_inventory_item_id",
                                    "variant_requires_shipping",
                                    "variant_inventory_management_used",
                                    "product_status",
                                    "product_published_at",
                                    "product_published_scope",
                                    "variant_max_updated_at"]]
shopify_products_sceleton.sort_values(["product_category_number","product_id","variant_sku","daydate"],ascending=[True,True,True,False]).reset_index(drop=True, inplace=True)


# In[33]:


# convert all dtypes
new_dict_keys = list(shopify_products_sceleton.columns)
dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
dict_dtypechanges_sceleton = dict_filter(dict_dtypechanges, new_dict_keys)
shopify_products_sceleton = shopify_products_sceleton.astype(dict_dtypechanges_sceleton)


# ## 4.3 Remove duplicates

# In[34]:


# in duplicate variant_id scenario take the one with lastest variant_max_updated_at

# get list of variant_id that are duplicate
duplicates = shopify_products_sceleton.groupby("variant_id", as_index=False).count()
duplicates = duplicates[duplicates["product_category"] > shopify_products_sceleton.daydate.nunique()]
duplicates = list(duplicates.variant_id)
duplicates

# split dataframe in duplicates and non duplicates
shopify_products_sceleton_duplicates = shopify_products_sceleton[shopify_products_sceleton["variant_id"].isin(duplicates)]
shopify_products_sceleton_nonduplicates = shopify_products_sceleton[~shopify_products_sceleton["variant_id"].isin(duplicates)]

# get only values for each duplicate with lastest variant_max_updated_at 
shopify_products_sceleton_duplicates.sort_values(by=["variant_id", "daydate", "variant_max_updated_at"], ascending=False, inplace=True)
shopify_products_sceleton_duplicates = shopify_products_sceleton_duplicates.groupby(["variant_id", "daydate"], as_index=False).first()

# concat again 
shopify_products_sceleton = pd.concat([shopify_products_sceleton_duplicates, shopify_products_sceleton_nonduplicates])
shopify_products_sceleton.sort_values(by=["product_category_number", "variant_id", "daydate"], inplace=True)
shopify_products_sceleton.reset_index(drop=True, inplace=True)


# # 5. Read data to database

# In[35]:


#params


# In[36]:


print("Sceleton Shape: ", shopify_products_sceleton.shape)
print("Sceleton maximum date is after transformations is: ", shopify_products_sceleton.daydate.max())


# In[37]:


#engine.dispose()


# In[38]:


#engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True

#engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False)

# In[39]:


t = Timer("Export")
shopify_products_sceleton.to_sql("shopify_productsxdays_sceleton", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
shopify_products.to_sql("shopify_products", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
# added multi channel sales
shopify_products_sceleton.to_sql("shopify_productsxdays_sceleton", con = engine, schema="sc_shopify", if_exists='replace', index=False, chunksize=1000, method="multi")
shopify_products.to_sql("product_info", con = engine, schema="sc_shopify", if_exists='replace', index=False, chunksize=1000, method="multi")
t.end()


# In[40]:


engine.dispose()


# In[41]:


print(params)

