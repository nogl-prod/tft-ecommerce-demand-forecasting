#!/usr/bin/env python
# coding: utf-8

# # TO DO

# - Dynamic CDC needed? -> Does Shopify always store non-active "old" products?
# - Transition merging of sub-tables into SQL queries for faster performance
# - Feature naming translation in the beginning, so code is resitant to raw data feature name changes
# - Add other tables: 2.1.3 - 2.1.7 and missing supply chain information
# - Add all "USE LATER" features
# - Add all "USE ELSEWHERE" features to other scripts

# # 0. Installs

# # 1. Imports/Options

# ## 1.1 External imports

# In[1]:


import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import STL


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

# In[4]:


pd.set_option('display.max_columns', None)


# # 2. Load data

# ## 2.1 Connect to DB

# # A initialization and configuration file is used to protect the author's login credentials
# 
# # Establish a connection to the database by creating a cursor object
# 
# # Obtain the configuration parameters
# params = config()
# 
# # Connect to the PostgreSQL database
# conn = psycopg2.connect(**params)
# # Create a new cursor
# cur = conn.cursor()

# ## 2.2 Load data in pandas dataframe

# ### 2.2.1 Shopify Sub-Table analysis

# #### 2.2.1.1 order - if needed add more info by levering airbyte_orders_hashid and combine with subtables:
# <font color='red'>- refunds --> DO NOT USE: Covered in seperate dataframe --> order_refunds</font><br>
# <font color='red'>- tax_lines --> DO NOT USE: Info already covered by existing feature order_total_tax and lineitem_taxlines_title/_rate</font><br>
# <font color='red'>- fulfillments --> DO NOT USE: Covered in seperate dataframe --> fulfillment_orders / fulfillments </font><br>
# <font color='red'>- total_tax_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='green'>- discount_codes --> USE: Extra info on type of discount </font><br>
# <font color='green'>- shipping_lines --> USE: Extra info on type of shipping (free, standard, extra...) </font><br>
# <font color='red'>- total_price_set --> DO NOT USE: No new information, shop vs. presentment money info already in columns</font><br>
# <font color='red'>- subtotal_price_set --> DO NOT USE: No new information, shop vs. presentment money info already in columns</font><br>
# <font color='red'>- total_discounts_set --> DO NOT USE: No new information, shop vs. presentment money info already in columns</font><br>
# <font color='red'>- discount_allocations --> DO NOT USE: Empty</font><br>
# <font color='red'>- current_total_tax_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='red'>- current_total_price_set --> DO NOT USE: No new information, shop vs. presentment money info already in columns</font><br>
# <font color='red'>- total_shipping_price_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='red'>- current_subtotal_price_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='red'>- total_line_items_price_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='red'>- current_total_discounts_set --> DO NOT USE: No new information, taxes already presented in shop currency</font><br>
# <font color='green'>- customer --> USE: Extra info number of new customers, avg. customer total spend </font><br>
# <font color='green'>- shipping_address --> USE: Extra info on geographical destination of order  </font><br>

# #### 2.2.1.2 line_items - if needed add more info by levering airbyte_line_items_hashid and combine with subtables:
# <font color='red'>- price_set --> DO NOT USE: No real difference to existing feature lineitem_price</font><br>
# <font color='green'>- tax_lines --> USE: tax rate, tax amount and country/title of tax used (e.g. MwSt)</font><br>
# <font color='red'>- total_discount_set --> DO NOT USE: No real difference to existing feature lineitem_total_discount</font><br>
# <font color='green'>- discount_allocations --> USE: discount amount allocated to line item and discount_allocation_index</font><br>
# <font color='green'>- properties --> USE: gives hint if it was a pre-order</font><br>

# #### 2.2.1.3 order_refunds / line_items (or orders_refunds subtable from orders) - if needed add more info by levering airbyte_orders_hashid and combine with subtables:
# - calculate return rates
# - does it make sense to do a time series forecast for refunds or rather a classification which lineitems per order will be refunded most likely? and when?

# #### 2.2.1.4 abandoned_checkouts / line_items - if needed add more info by levering airbyte_orders_hashid and combine with subtables:
# - calculate total abandoned checkout over all products to indicate short term demand potential
# - calculate number of times item was abandoned per day
# - calculate average order values/discounts

# #### 2.2.1.5 discount_codes - if needed add more info by levering airbyte_orders_hashid and combine with subtables:
# -
# 
# --> Can be plan data

# #### 2.2.1.6 price_rules - if needed add more info by levering airbyte_orders_hashid and combine with subtables:
# -
# 
# --> Can be plan data

# #### 2.2.1.7 customer table as well --> ATTENTION: Data privacy rights...
# 
# --> Analyse cross-selling potentional with last_order_id: e.g. how many clients order again after ordering one specific sku?
# --> Feature: number of existing customers
# --> orders_customer_default_address country um nachzuvollziehen wo der purchase getätigt wurde

# #### MISSING SUPPLY CHAIN INFO:
# - inventory_levels --> In products script?
# - fulfillment_orders / fulfillments /orders_fulfillment (from orders table)

# ### 2.2.2 Load selected dataframes 

# In[5]:


# sceleton data frame
shopify_productxdays_sceleton = import_data_AWSRDS(schema="transformed",table="shopify_productsxdays_sceleton",engine=engine)

# products data frame
shopify_products = import_data_AWSRDS(schema="transformed",table="shopify_products",engine=engine)

# main dataframes
schema_main = "shopify"

shopify_lineitems = import_data_AWSRDS(schema=schema_main,table="orders_line_items",engine=engine)
shopify_lineitems = shopify_lineitems[["_airbyte_line_items_hashid", 
                                       "_airbyte_orders_hashid", 
                                       "id", 
                                       "sku",
                                       "product_id",
                                       "variant_id",
                                       "product_exists",
                                       "quantity",
                                       "price",
                                       "total_discount",
                                       "taxable", 
                                       "gift_card",
                                       "requires_shipping",
                                       "fulfillment_status",
                                       "fulfillment_service",
                                       "fulfillable_quantity",
                                       "variant_inventory_management"]]

shopify_orders = import_data_AWSRDS(schema=schema_main,table="orders",engine=engine)
shopify_orders = shopify_orders[["_airbyte_orders_hashid", 
                                 "id",
                                 "order_number",
                                 "number",
                                 "test",
                                 "gateway",
                                 "confirmed",
                                 "closed_at",
                                 "created_at",
                                 "updated_at",
                                 "cancelled_at",
                                 "processed_at",
                                 "currency",
                                 "presentment_currency",
                                 "total_price",
                                 "subtotal_price", 
                                 "current_subtotal_price",
                                 "total_line_items_price",
                                 "total_tip_received",
                                 "current_total_price",
                                 "total_price_usd", 
                                 "financial_status",
                                 "total_discounts",
                                 "current_total_discounts",
                                 "current_total_tax",
                                 "total_tax",
                                 "taxes_included",
                                 "customer_locale",
                                 "source_name",
                                 "source_identifier",
                                 "landing_site",
                                 "referring_site",
                                 "buyer_accepts_marketing",
                                 "total_weight",
                                 "processing_method",
                                 "total_outstanding",
                                 "fulfillment_status"]]

# sub-tables of lineitems:
shopify_lineitems_discount_allocations = import_data_AWSRDS(schema=schema_main,table="orders_line_items_discount_allocations",engine=engine)
shopify_lineitems_discount_allocations = shopify_lineitems_discount_allocations[["_airbyte_line_items_hashid", "amount", "discount_application_index"]]

shopify_lineitems_tax_lines = import_data_AWSRDS(schema=schema_main,table="orders_line_items_tax_lines",engine=engine)
shopify_lineitems_tax_lines = shopify_lineitems_tax_lines[["_airbyte_line_items_hashid", "rate", "price", "title"]]

shopify_lineitems_properties = import_data_AWSRDS(schema=schema_main,table="orders_line_items_properties",engine=engine)
shopify_lineitems_properties = shopify_lineitems_properties[["_airbyte_line_items_hashid","name"]]

# sub-tables of order:
shopify_orders_discount_codes = import_data_AWSRDS(schema=schema_main,table="orders_discount_codes",engine=engine)
shopify_orders_discount_codes = shopify_orders_discount_codes[["_airbyte_orders_hashid", "code", "type"]]

shopify_orders_shipping_lines = import_data_AWSRDS(schema=schema_main,table="orders_shipping_lines",engine=engine)
shopify_orders_shipping_lines = shopify_orders_shipping_lines[["_airbyte_orders_hashid", "id", "code", "price", "title", "source", "discounted_price"]]

shopify_orders_customer = import_data_AWSRDS(schema=schema_main,table="orders_customer",engine=engine)
shopify_orders_customer = shopify_orders_customer[["_airbyte_orders_hashid", "total_spent", "orders_count"]]

shopify_orders_shipping_address = import_data_AWSRDS(schema=schema_main,table="orders_shipping_address",engine=engine)
shopify_orders_shipping_address = shopify_orders_shipping_address[["_airbyte_orders_hashid", "country_code"]]


# # Utilize the create_pandas_table function to create a Pandas data frame
# # Store the data as a variable
# 
# # sceleton data frame
# shopify_productxdays_sceleton = create_pandas_table(sql_productxdays_sceleton, database = conn)
# 
# # products data frame
# shopify_products = create_pandas_table(sql_products, database = conn)
# 
# # main dataframes
# shopify_lineitems = create_pandas_table(sql_lineitems, database = conn)
# shopify_orders = create_pandas_table(sql_orders, database = conn)
# 
# # sub-tables of lineitems:
# shopify_lineitems_discount_allocations = create_pandas_table(sql_lineitems_discount_allocations, database = conn)
# shopify_lineitems_tax_lines = create_pandas_table(sql_lineitems_tax_lines, database = conn)
# shopify_lineitems_properties = create_pandas_table(sql_lineitems_properties, database = conn)
# 
# # sub-tables of order:
# shopify_orders_discount_codes = create_pandas_table(sql_orders_discount_codes, database = conn)
# shopify_orders_shipping_lines = create_pandas_table(sql_orders_shipping_lines, database = conn)
# shopify_orders_customer = create_pandas_table(sql_orders_customer, database = conn)
# shopify_orders_shipping_address = create_pandas_table(sql_orders_shipping_address, database = conn)

# # 3. Transform & clean data

# ## 3.1 Merge Shopify dataframes

# ### 3.1.1 Merge subtables of line_items to line_items table using _airbyte_line_items_hashid

# In[6]:


# rename columns with source prefix to keep track of column source
rename_dfcolumns(shopify_lineitems_discount_allocations, "discountallocations_")
rename_dfcolumns(shopify_lineitems_tax_lines, "taxlines_")
rename_dfcolumns(shopify_lineitems_properties, "properties_")

# merge sub-tables to lineitems table
for d in [shopify_lineitems_discount_allocations, shopify_lineitems_tax_lines, shopify_lineitems_properties]:
    shopify_lineitems = shopify_lineitems.merge(d, how="left", on="_airbyte_line_items_hashid")

shopify_lineitems.drop_duplicates(inplace=True) # drop duplicate rows


# ### 3.1.2 Merge subtables of orders to orders table using _airbyte_orders_hashid

# In[7]:


# rename columns with source prefix to keep track of column source
rename_dfcolumns(shopify_orders_discount_codes, "discountcodes_")
rename_dfcolumns(shopify_orders_shipping_lines, "shippinglines")
rename_dfcolumns(shopify_orders_customer, "customer_")
rename_dfcolumns(shopify_orders_shipping_address, "shippingaddress_")

# merge sub-tables to lineitems table
for d in [shopify_orders_discount_codes, shopify_orders_shipping_lines, shopify_orders_customer, shopify_orders_shipping_address]:
    shopify_orders = shopify_orders.merge(d, how="left", on="_airbyte_orders_hashid")
    
shopify_orders.drop_duplicates(inplace=True) # drop duplicate rows


# ### 3.1.3 Merge orders table on line_items table

# In[8]:


# rename columns for each dataframe to keep track of source
rename_dfcolumns(shopify_orders,"orders_")
rename_dfcolumns(shopify_lineitems,"lineitems_")

# merge and create shopify_sales dataframe
shopify_sales_raw = shopify_lineitems.merge(shopify_orders, how="left", on="_airbyte_orders_hashid")


# ## 3.2 Transform data

# ### 3.2.1 Analyse transformation / feature engineering to be done
# <font color='red'>- DELETE: _airbyte_line_items_hashid</font><br>
# <font color='red'>- DELETE: _airbyte_orders_hashid</font><br>
# #### DATA FROM shopify.orders_line_items
# <font color='red'>- DELETE: lineitems_id</font><br>
# <font color='green'>- KEEP: lineitems_sku --> AGGREGATION: GROUPING LEVEL or lineitems_variant_id</font><br>
# <font color='green'>- KEEP: lineitems_product_id --> "The ID of the product that the line item belongs to. Can be null if the original product associated with the order is deleted at a later date."</font><br>
# <font color='green'>- KEEP: lineitems_variant_id --> AGGREGATION: GROUPING LEVEL or lineitems_sku</font><br>
# <font color='red'>- DELETE: lineitems_product_exists -> "It indicates that the original product is deleted"</font><br>
# <font color='blue'>- TRANSFORM: lineitems_quantity --> AGGREGATION: Transform into target variable, using sum() over quantities on sku and daily level </font><br>
# <font color='blue'>- TRANSFORM: lineitems_price --> AGGREGATION: Daily average</font><br>
# <font color='red'>- DELETE: lineitems_total_discount --> "The total amount of the discount allocated to the line item in the shopcurrency. This field must be explicitly set using draft orders, Shopify scripts, or the API. Instead of usingthis field, Shopify recommends using discount_allocations, which provides the same information." --> Covered by discount_allocations</font><br>
# <font color='red'>- DELETE/REDUNDANT: lineitems_taxable --> Static in products table</font><br>
# <font color='green'>- KEEP: lineitems_gift_card</font><br>
# <font color='red'>- DELETE/REDUNDANT: lineitems_requires_shipping --> Static in products table</font><br>
# <font color='#F535AA'>- USE ELSEWHERE: lineitems_fulfillment_status @order_refunds & @inventory --> Maybe use here?</font><br>
# <font color='#F535AA'>- USE ELSEWHERE: lineitems_fulfillment_service @order_refunds & @inventory --> Maybe use here?</font><br>
# <font color='#F535AA'>- USE ELSEWHERE: lineitems_fulfillable_quantity @order_refunds & @inventory --> Maybe use here?</font><br>
# <font color='blue'>- TRANSFORM: lineitems_variant_inventory_management --> Boolean if inventory was tracked or not, check static from products table also --> AGGREGATION: Should yield the same</font><br>
# <font color='blue'>- TRANSFORM: lineitems_discountallocations_amount --> "The discount amount allocated to the line in the shop currency" --> Daily average discount in percent of average price --> AGGREGATION: Daily average</font><br>
# <font color='red'>- DELETE: lineitems_discountallocations_discount_application_index --> "The index of the associated discount application in the order'sdiscount_applications"</font><br>
# <font color='blue'>- TRANSFORM: lineitems_taxlines_rate --> AGGREGATION: Daily average tax rate</font><br>
# <font color='blue'>- TRANSFORM: lineitems_taxlines_price --> AGGREGATION: Daily average tax</font><br>
# <font color='red'>- DELETE: lineitems_taxlines_title --> Cannot be averaged</font><br>
# #### DATA FROM shopify.orders
# <font color='blue'>- TRANSFORM: orders_id --> AGGREGATION: Calculate daily number of orders sku was involved in</font><br>
# <font color='red'>- DELETE: orders_order_number</font><br>
# <font color='red'>- DELETE: orders_number</font><br>
# <font color='blue'>- TRANSFORM: orders_test --> Remove test orders from dataframe --> AGGREGATION: Deleted in Transformation</font><br>
# <font color='blue'>- TRANSFORM: orders_gateway --> one hot encoded categorical --> AGGREGATION: use mean to calcualte shares of each category</font><br>
# <font color='red'>- DELETE: orders_confirmed</font><br>
# <font color='blue'>- TRANSFORM: orders_closed_at --> AGGREGATION: Calculate average time order creation to closing per sku</font><br>
# <font color='blue'>- TRANSFORM: orders_created_at --> AGGREGATION: GROUPING LEVEL Use as day order was made by customer</font><br>
# <font color='red'>- DELETE: orders_updated_at</font><br>
# <font color='blue'>- TRANSFORM: orders_cancelled_at --> Convert to boolean if order was cancelled --> AGGREGATION: Calculate percentage of number of orders cancelled involving sku per day</font><br>
# <font color='blue'>- TRANSFORM: orders_processed_at --> AGGREGATION: Calculate average time order creation to processing and processing to closing per sku</font><br>
# <font color='blue'>- TRANSFORM: orders_currency --> Convert to category one hot encoding --> AGGREGATION: Share of transactions in EUR</font><br>
# <font color='blue'>- TRANSFORM: orders_presentment_currency --> Convert to category one hot encoding --> AGGREGATION: Share of transactions in EUR</font><br>
# <font color='blue'>- TRANSFORM: orders_total_price --> Calculate daily/weekly/monthly average total price of orders including this sku</font><br>
# <font color='blue'>- TRANSFORM: orders_current_total_price --> "The current total price of the order in the shop currency. The value of this field reflects order edits, returns, and refunds." --> Calculate daily/weekly/monthly average percentage of total_price of orders including this sku</font><br>
# <font color='blue'>- TRANSFORM: orders_subtotal_price --> "The subtotal price is the list price after discounts without shipping and taxes, and the total price includes shipping and taxes." --> Calculate daily/weekly/monthly average subtotal price of orders including this sku</font><br>
# <font color='blue'>- TRANSFORM: orders_current_subtotal_price --> "The current subtotal price of the order in the shop currency. The value of this field reflects order edits, returns, and refunds." --> Calculate daily/weekly/monthly average percentage of subtotal_price of orders including this sku</font><br> 
# <font color='blue'>- TRANSFORM: orders_total_line_items_price --> From Analysis: Compared to orders_total_price this one still includes discounts --> Calculate daily/weekly/monthly average total line_item_price of orders including this sku</font><br>
# <font color='red'>- DELETE: orders_total_tip_received</font><br>
# <font color='red'>- DELETE: orders_total_price_usd</font><br>
# <font color='#F535AA'>- USE ELSEWHERE: orders_financial_status --> @order_refunds use "partially refunded" here?</font><br>
# <font color='blue'>- TRANSFORM: orders_total_discounts --> "The total discounts applied to the price of the order in the shop currency." --> Calculate daily/weekly/monthly average percentage of orders_total_price / orders_total_line_items_price (rather this one, as orders_total_price seems to have discounts already subtracted)</font><br>
# <font color='blue'>- TRANSFORM: orders_current_total_discounts --> "The current total discounts on the order in the shop currency. The value of this field reflects order edits, returns, and refunds" Calculate daily/weekly/monthly average percentage of orders_current_total_price (add discount before, as it seems to not be included)</font><br>
# <font color='red'>- DELETE: orders_total_tax</font><br>
# <font color='red'>- DELETE: orders_current_total_tax --> "The current total taxes charged on the order in the shop currency. The value of this field reflects order edits, returns, or refunds."</font><br>
# <font color='red'>- DELETE: orders_taxes_included</font><br>
# <font color='blue'>- TRANSFORM: orders_customer_locale --> One hot encoding AGGREGATION: Calculate shares per country / region</font><br>
# <font color='blue'>- TRANSFORM: orders_source_name --> Convert to boolean with web = True, rest = False --> AGGREGATION: Share of web traffic </font><br>
# <font color='red'>- DELETE: orders_source_identifier --> Do not know what </font><br>
# <font color='#F535AA'>- USE LATER: orders_landing_site --> Transform and leverage low funnel traffic information (split different landing sites)</font><br>
# <font color='#F535AA'>- USE LATER: orders_referring_site --> Transform and leverage medium funnel traffic information (split different reffering sites)</font><br>
# <font color='blue'>- TRANSFORM: orders_buyer_accepts_marketing --> Convert to boolean --> AGGREGATION: Calculate average percentage of buyer accepting marketing </font><br>
# <font color='blue'>- TRANSFORM: orders_total_weight --> AGGREGATION: Calculate average weight of total orders</font><br>
# <font color='blue'>- TRANSFORM: orders_processing_method --> Convert to category one hot encoding --> AGGREGATION: Calculate average percentage shares per category</font><br>
# <font color='red'>- DELETE: orders_total_outstanding</font><br>
# <font color='#F535AA'>- USE ELSEWHERE: orders_fulfillment_status --> Use in supply chain tables? e.g. tracking inventory levels with blocked supply for outgoing quantites @inventory_levels</font><br>
# <font color='red'>- USE ELSEWHERE: orders_discountcodes_code --> @discount_codes</font><br>
# <font color='blue'>- TRANSFORM: orders_discountcodes_type --> Convert to category one hot encoding --> AGGREGATION: Calculate average percentage shares per category</font><br>
# <font color='red'>- DELETE: orders_shippinglinesid</font><br>
# <font color='#F535AA'>- USE LATER:: orders_shippinglinescode --> Add feature covering active shipping options and prices</font><br>
# <font color='blue'>- TRANSFORM: orders_shippinglinesprice --> Calculate daily/weekly/monthly average shipping price</font><br>
# <font color='red'>- DELETE/REDUNDANT: orders_shippinglinestitle --> Same as orders_shippinglinescode</font><br>
# <font color='red'>- DELETE: orders_shippinglinessource --> All "shopify" or NaN</font><br>
# <font color='blue'>- TRANSFORM: orders_shippinglinesdiscounted_price --> Calculate daily/weekly/monthly average percentage of orders_shippinglinesprice</font><br>
# <font color='blue'>- TRANSFORM: orders_customer_total_spent --> AGGREGATION: Calculate average customer total customer spending for orders including that sku</font><br>
# <font color='blue'>- TRANSFORM: orders_customer_orders_count --> Convert to boolean for >1 False AGGREGATION: Calculate share of new customers ordering that product</font><br>

# In[9]:


# load in only features defined as "KEEP" or "TRANSFORM" from analysis
shopify_sales_transform = shopify_sales_raw[["lineitems_sku", 
                                           "lineitems_product_id", 
                                           "lineitems_variant_id", 
                                           "lineitems_quantity", 
                                           "lineitems_price", 
                                           "lineitems_gift_card", 
                                           "lineitems_variant_inventory_management", 
                                           "lineitems_discountallocations_amount",
                                           "lineitems_taxlines_rate",
                                           "lineitems_taxlines_price",
                                           "lineitems_properties_name",
                                           "orders_id",
                                           "orders_test", 
                                           "orders_gateway",
                                           "orders_closed_at",
                                           "orders_created_at", 
                                           "orders_cancelled_at",
                                           "orders_processed_at",
                                           "orders_currency",
                                           "orders_presentment_currency",
                                           "orders_total_price",
                                           "orders_current_total_price",
                                           "orders_subtotal_price",
                                           "orders_current_subtotal_price",
                                           "orders_total_line_items_price",
                                           "orders_total_discounts",
                                           "orders_current_total_discounts",
                                           "orders_customer_locale",
                                           "orders_source_name",
                                           #"orders_landing_site",
                                           #"orders_referring_site",
                                           "orders_buyer_accepts_marketing",
                                           "orders_total_weight",
                                           "orders_processing_method",
                                           "orders_discountcodes_type",
                                           "orders_shippinglinesprice",
                                           "orders_shippinglinesdiscounted_price",
                                           "orders_customer_total_spent",
                                           "orders_customer_orders_count",
                                           "orders_shippingaddress_country_code"]]


# In[10]:


# define dtypes and cleaning procedures per feature: 
shopify_sales_cleaning_1 = pd.DataFrame([["lineitems_sku","str", "", "get lineitems_sku from shopify.products table via variant_id"], 
                                       ["lineitems_product_id","int64", 0, "get lineitems_product_id from shopify.products table via variant_id/variant_sku"], 
                                       ["lineitems_variant_id","int64", 0, "get lineitems_variant_id from shopify.products table via variant_sku"], 
                                       ["lineitems_quantity","float64", 0, "SKIP"], 
                                       ["lineitems_price","float64", "bfill", "SKIP"], # bfill because sorting descending with date
                                       ["lineitems_gift_card","boolean", False, "SKIP"],
                                       ["lineitems_variant_inventory_management","boolean", False, "replace with {""lineitems_variant_inventory_management"":[[""shopify"",None],[True, False]]}"], # with replace() converted into boolean
                                       ["lineitems_discountallocations_amount","float64", 0, "SKIP"],
                                       ["lineitems_taxlines_rate","float64", 0, "use taxrate averages per country and merge this data in"],
                                       ["lineitems_taxlines_price","float64", 0, "(lineitems_price - lineitems_discountallocations_amount) - (lineitems_price - lineitems_discountallocations_amount) * 1/(1+lineitems_taxlines_rate)"],
                                       ["lineitems_properties_name","boolean",False,"SKIP"], # feature here with different name, as later it is renamed
                                       ["orders_id","int64", 0, "SKIP"], 
                                       ["orders_test","boolean", False, "delete those rows and drop feature"],
                                       ["orders_gateway","category", "Unknown", "consolidieren von Klarna und one hot encoding"],
                                       ["orders_closed_at", "float", np.nan, "individual date transformation to int with difference to created_at"], # feature here with different name, as later it is renamed
                                       ["orders_created_at", "date", "SKIP", "individual date transformation with pd.to_datetime(utc=True)"],
                                       ["orders_cancelled_at", "boolean", False, "conversion to boolean"], # feature here with different name, as later it is renamed
                                       ["orders_processed_at", "float", np.nan, "individual date transformation to int with difference to created_at"], # feature here with different name, as later it is renamed
                                       ["orders_currency","boolean", "SKIP", "make EUR = True rest = False"], # feature here with different name, as later it is renamed
                                       ["orders_presentment_currency","boolean", "SKIP", "make EUR = True rest = False"], # feature here with different name, as later it is renamed
                                       ["orders_total_price","float64", np.nan, "AFTER AGGREGATION"], 
                                       ["orders_current_total_price","float64", np.nan, "AFTER AGGREGATION"],
                                       ["orders_subtotal_price","float64", np.nan, "AFTER AGGREGATION"],
                                       ["orders_current_subtotal_price","float64", np.nan, "AFTER AGGREGATION"],
                                       ["orders_total_line_items_price","float64", np.nan, "AFTER AGGREGATION"],
                                       ["orders_total_discounts","float64", np.nan, "SKIP"],
                                       ["orders_current_total_discounts","float64", np.nan, "SKIP"],
                                       ["orders_customer_locale","category", "Unknown", "simplification by language origin and one hot encoding"],
                                       ["orders_source_name","boolean",False,"Convert to boolean with web = True, rest = False"], # feature here with different name, as later it is renamed
                                       #["orders_landing_site","str", "SKIP", "extract lower funnel information"],
                                       #["orders_referring_site","str", "SKIP", "extract higher funnel information"],
                                       ["orders_buyer_accepts_marketing","boolean", False, "SKIP"],
                                       ["orders_total_weight","float64","SKIP", "SKIP"],
                                       ["orders_processing_method","category","Unknown", "one hot encoding"],
                                       ["orders_discountcodes_type","category", "Unknown", "one hot encoding"],
                                       ["orders_shippinglinesprice","float64", 0, "SKIP"],
                                       ["orders_shippinglinesdiscounted_price","float64", 0, "SKIP"],
                                       ["orders_customer_total_spent","float64",np.nan, "SKIP"],
                                       ["orders_customer_orders_count","boolean", True , "Convert to boolean for >1 False"], # feature here with different name, as later it is renamed
                                       ["orders_shippingaddress_country_code","category","DE","SKIP"]], 
                                        columns=["feature","dtype","fillna_individual", "fillna_special"]) 
# add old dtypes
shopify_sales_cleaning_1 = shopify_sales_cleaning_1.merge(shopify_sales_raw.dtypes.reset_index().rename(columns={"index":"feature",0:"raw_dtype"}), how="left", on="feature")

# add number of nan values to each feature
shopify_sales_cleaning_1 = shopify_sales_cleaning_1.merge(pd.DataFrame(shopify_sales_transform.isna().sum()).reset_index().rename(columns={0:"number_of_NaN_values","index":"feature"}), how="left", on="feature")

#add number of 0s to each feature
zero_df = pd.DataFrame(columns=["feature", "number_of_zeros"])
for c in list(shopify_sales_transform.columns):
    zero_df = pd.concat([zero_df,pd.DataFrame([[c,(shopify_sales_transform[c] == 0).sum()]],columns=["feature", "number_of_zeros"])])
shopify_sales_cleaning_1 = shopify_sales_cleaning_1.merge(zero_df, how="left",on="feature")

# change order of columns
shopify_sales_cleaning_1 = shopify_sales_cleaning_1[['feature', 'dtype', 'raw_dtype', 'fillna_individual', 'fillna_special',
       'number_of_NaN_values', 'number_of_zeros']]

# rename features that are renamend in the process of this script
shopify_sales_cleaning_1["feature"].replace({"orders_closed_at":"orders_closed_after",
                                         "orders_cancelled_at":"orders_cancelled",
                                         "orders_processed_at":"orders_processed_after",
                                         "orders_currency":"orders_currency_EUR",
                                         "orders_presentment_currency":"orders_presentment_currency_EUR",
                                         "orders_source_name":"orders_source_web",
                                         "orders_customer_orders_count":"orders_customer_orders_new",
                                         "lineitems_properties_name":"lineitems_preorder"}, inplace=True)
shopify_sales_cleaning_1


# ### 3.2.2 Clean #round1/ transform before heavy transformation and feature recalculation

# In[11]:


# drop duplicates
shopify_sales_transform.drop_duplicates(inplace=True)


# #### 3.2.2.1 numerics "" replacement and dtype changes (float)

# In[12]:


# convert "" to NaN for all numeric
list_numeric_columns_to_replace = list(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "int64") | (shopify_sales_cleaning_1["dtype"] == "float64")].feature)
for c in list_numeric_columns_to_replace:
    shopify_sales_transform[c].replace("",np.nan, inplace=True)

# convert numercis to needed dtypes

# replace all NaN values with 0s
for i in list(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "int64")].feature):
    shopify_sales_transform[i].replace(np.nan, 0, inplace=True)

# convert to final dtypes
dict_dtypechanges_numerics = pd.Series(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "int64") | (shopify_sales_cleaning_1["dtype"] == "float64")].dtype.values,index=shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "int64") | (shopify_sales_cleaning_1["dtype"] == "float64")].feature).to_dict()
shopify_sales_transform = shopify_sales_transform.astype(dict_dtypechanges_numerics)


# #### 3.2.2.2 date columns transformation (fillna_special for date dtypes)

# In[13]:


# change dtypes
def change_dtype_dates(df, list_of_columns,utc_true=True):
    for c in list_of_columns:
        df[c] = pd.to_datetime(df[c], utc=utc_true)

change_dtype_dates(shopify_sales_transform, ["orders_closed_at", "orders_processed_at", "orders_created_at"])

# calculate differences to orders_created_at
shopify_sales_transform["orders_closed_after"] = (shopify_sales_transform["orders_closed_at"]- shopify_sales_transform["orders_created_at"]).dt.total_seconds()/60
shopify_sales_transform["orders_processed_after"] = (shopify_sales_transform["orders_processed_at"] - shopify_sales_transform["orders_created_at"]).dt.total_seconds()/60
shopify_sales_transform.drop(columns=["orders_closed_at","orders_processed_at"], inplace=True)

# transformation of orders_cancelled_at see booleans

# transform dates to daily format as later it will be aggregated on that level
shopify_sales_transform["orders_created_at"] = pd.to_datetime(shopify_sales_transform["orders_created_at"], utc=True).dt.date


# #### 3.2.2.3 sort dataframe

# In[14]:


# sort dataframe
shopify_sales_transform = shopify_sales_transform.sort_values(["lineitems_product_id","lineitems_sku","orders_created_at"],ascending=[True,True,False]).reset_index(drop=True)


# #### 3.2.2.4 get id's from shopify.products table (fillna_special for lineitem_sku, lineitem_product_id and lineitem_variant_id)

# In[15]:


# get lineitems_sku from shopify.products table via variant_id
shopify_sales_transform = shopify_sales_transform.merge(shopify_products[["variant_id", "variant_sku"]].astype({"variant_id":"int64","variant_sku":"str"}), how="left", left_on="lineitems_variant_id", right_on="variant_id")
shopify_sales_transform = shopify_sales_transform.sort_values(["lineitems_product_id","lineitems_sku","orders_created_at"],ascending=[True,True,False]).reset_index(drop=True)
shopify_sales_transform["lineitems_sku"].replace(0, np.nan, inplace=True) # NaNs were previously replaced with 0s so conversion to int64 and then merging is successful
shopify_sales_transform["lineitems_sku"].fillna(shopify_sales_transform["variant_sku"], inplace=True)
shopify_sales_transform["lineitems_sku"].replace(np.nan, 0, inplace=True) # change back to 0s
shopify_sales_transform.drop(columns=["variant_id", "variant_sku"], inplace=True) # drop merged columns

# get lineitems_product_id and lineitems_variant_id from shopify.products table via variant_sku
shopify_sales_transform = shopify_sales_transform.merge(shopify_products[["variant_sku", "variant_id", "product_id"]], how="left", left_on="lineitems_sku", right_on="variant_sku")
shopify_sales_transform = shopify_sales_transform.sort_values(["lineitems_product_id","lineitems_sku","orders_created_at"],ascending=[True,True,False]).reset_index(drop=True)
shopify_sales_transform["lineitems_variant_id"].replace(0, np.nan, inplace=True) # NaNs were previously replaced with 0s so conversion to int64 and then merging is successful
shopify_sales_transform["lineitems_product_id"].replace(0, np.nan, inplace=True)
shopify_sales_transform["lineitems_variant_id"].fillna(shopify_sales_transform["variant_id"], inplace=True)
shopify_sales_transform["lineitems_product_id"].fillna(shopify_sales_transform["product_id"], inplace=True)
shopify_sales_transform["lineitems_variant_id"].replace(np.nan, 0, inplace=True) # change back to 0s
shopify_sales_transform["lineitems_product_id"].replace(np.nan, 0, inplace=True) 
shopify_sales_transform.drop(columns=["variant_sku", "variant_id", "product_id"], inplace=True) # drop merged columns


# #### 3.2.2.5 boolean columns

# dict_of_boolean_transformations = {"lineitems_variant_inventory_management":[["shopify",None],[True, False]]}
# # execute special boolean transformations with replace and raise error if more than two (True and False) entries in column
# for b in dict_of_boolean_transformations:
#     shopify_sales_transform[b] = shopify_sales_transform[b].replace(dict_of_boolean_transformations.get(b)[0],dict_of_boolean_transformations.get(b)[1])

# In[16]:


# SPECIAL TRANSFORMS

# create dict of all special boolean transformations with replace
shopify_sales_transform["lineitems_variant_inventory_management"] = shopify_sales_transform["lineitems_variant_inventory_management"].apply(lambda x: True if x != "" else False)

# make orders_currency and orders_presentment_currency a boolean with "EUR" = True and the rest being false
shopify_sales_transform["orders_currency"] = (shopify_sales_transform["orders_currency"] == "EUR")
shopify_sales_transform["orders_presentment_currency"] = (shopify_sales_transform["orders_presentment_currency"] == "EUR")
shopify_sales_transform.rename(columns={"orders_currency":"orders_currency_EUR","orders_presentment_currency":"orders_presentment_currency_EUR"}, inplace=True) # rename
# make orders_cancelled_at a boolean with NaN = False and rest being True
shopify_sales_transform["orders_cancelled_at"] = (pd.isna(shopify_sales_transform["orders_cancelled_at"]) == False)
shopify_sales_transform.rename(columns={"orders_cancelled_at":"orders_cancelled"}, inplace=True) # rename
# make orders_source_name a boolean with "web" = True and rest being False
shopify_sales_transform["orders_source_name"] = (shopify_sales_transform["orders_source_name"] == "web")
shopify_sales_transform.rename(columns={"orders_source_name":"orders_source_web"}, inplace=True) # rename
# make orders_customer_orders_count a boolean with >1 = False and rename orders_customer_new
shopify_sales_transform["orders_customer_orders_count"] = ((shopify_sales_transform["orders_customer_orders_count"] <= 1)|(pd.isna(shopify_sales_transform["orders_customer_orders_count"])==True))
shopify_sales_transform.rename(columns={"orders_customer_orders_count":"orders_customer_orders_new"}, inplace=True) # rename
# make lineitems_properties_name a boolean with checking for Preorders
shopify_sales_transform["lineitems_properties_name"] = shopify_sales_transform["lineitems_properties_name"].str.contains("Preorder")
shopify_sales_transform.rename(columns={"lineitems_properties_name":"lineitems_preorder"}, inplace=True) # rename

# CLEANING WITH FILLNA
# create dict of all boolean columns to clean with fillna
dict_boolean_fillna = pd.Series(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "boolean")].fillna_individual.values,index=shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "boolean")].feature).to_dict()
# fill in fillna_individuals for boolean columns with NaN values
shopify_sales_transform.fillna(value=dict_boolean_fillna, inplace=True)

# DTYPE CHANGES
# change dtypes of all boolean columns
dict_boolean_dtypes = pd.Series(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "boolean")].dtype.values,index=shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "boolean")].feature).to_dict()
shopify_sales_transform = shopify_sales_transform.astype(dict_boolean_dtypes)

# CHECK FOR ERRORS
# check if transformation and cleaning without errors
for b in dict_boolean_dtypes:
    if len(shopify_sales_transform[b].unique()) > 2:
        raise NameError("Boolean transformation not successful for: "+b)


# #### 3.2.2.6 fillna for rest

# In[17]:


dict_rest_fillna = pd.Series(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["fillna_individual"] != "SKIP")].fillna_individual.values,index=shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["fillna_individual"] != "SKIP")].feature).to_dict()
shopify_sales_transform.fillna(value=dict_rest_fillna, inplace=True)


# #### 3.2.2.7 dtype changes

# In[18]:


# convert to needed dtypes
dict_dtypechanges = pd.Series(shopify_sales_cleaning_1.loc[shopify_sales_cleaning_1["dtype"] != "date"].dtype.values,index=shopify_sales_cleaning_1.loc[shopify_sales_cleaning_1["dtype"] != "date"].feature).to_dict()
shopify_sales_transform = shopify_sales_transform.astype(dict_dtypechanges)


# #### 3.2.2.8 remove test orders from data and drop orders_test column

# In[19]:


shopify_sales_transform  = shopify_sales_transform.loc[shopify_sales_transform["orders_test"] == False].copy()
shopify_sales_transform.drop(columns=["orders_test"], inplace=True) # no longer needed


# ### 3.2.3 Heavy transformation and feature recalculation to prepare for aggregation on day x sku level (incl. fillna_special & clean #round2)

# #### 3.2.3.1 Funnel data extraction - DO LATER

# orders_referring_site
# substrings_to_replace = ['^.*?(?=.)']
# 
# for s in substrings_to_replace:
#     shopify_sales_transform['orders_referring_site'].replace(to_replace=s, value='', regex=True, inplace=True)

# #### 3.2.3.2 Prices

# This is done after aggregation, as filling now and then averaging will reduce accuracy of filling NaN or 0 values

# #### 3.2.3.3 Discounts

# calculate percentages
shopify_sales_transform["lineitems_discountallocations_amount"] = shopify_sales_transform["lineitems_discountallocations_amount"]/(shopify_sales_transform["lineitems_price"]*shopify_sales_transform["lineitems_quantity"]) # calculate percentage of discounted price 
shopify_sales_transform["lineitems_discountallocations_amount"].fillna(0, inplace=True) # fill NaN with 0
shopify_sales_transform["lineitems_discountallocations_amount"].replace([np.inf, -np.inf], 0, inplace=True) # replace infinite values with 0 because we devided by 0 price

shopify_sales_transform["orders_total_discounts"] = shopify_sales_transform["orders_total_discounts"]/shopify_sales_transform["orders_total_line_items_price"] # calculate percentage of orders_total_line_items_price as in orders_total_price the discount is already substracted
shopify_sales_transform["orders_total_discounts"].fillna(0, inplace=True) # fill NaN with 0

shopify_sales_transform["orders_current_total_discounts"] = shopify_sales_transform["orders_current_total_discounts"]/(shopify_sales_transform["orders_current_total_price"]+shopify_sales_transform["orders_current_total_discounts"]) # calculate percentage of orders_current_total_price + orders_current_total_discounts as discount already is substracted from orders_current_total_price
shopify_sales_transform["orders_current_total_discounts"].fillna(0, inplace=True) # fill NaN with 0

# #### 3.2.3.4 Taxes

# TAX RATES
# create table featuring average tax rates per country
tax_allocation_table = shopify_sales_transform[["lineitems_taxlines_rate", "orders_customer_locale"]].groupby("orders_customer_locale", as_index=False).agg(pd.Series.mode).replace(to_replace=[],value=0)
tax_allocation_table.lineitems_taxlines_rate = tax_allocation_table.lineitems_taxlines_rate.apply(lambda x: 0 if (type(x)==np.ndarray) else x)

# merge tax rates on empty values
tax_allocation_table.rename(columns={"lineitems_taxlines_rate":"TEMP_lineitems_taxlines_rate"}, inplace=True) # rename for merge
shopify_sales_transform = shopify_sales_transform.merge(tax_allocation_table, how="left", on="orders_customer_locale")
shopify_sales_transform

# prepare tax_temp_df for fillna and execute, replacing NaN values in lineitems_taxlines_rate with new tax rates
shopify_sales_transform = shopify_sales_transform.sort_values(["lineitems_product_id","lineitems_sku","orders_created_at"],ascending=[True,True,False]).reset_index(drop=True) # sort values important for fillna
shopify_sales_transform["lineitems_taxlines_rate"].fillna(shopify_sales_transform["TEMP_lineitems_taxlines_rate"], inplace=True)
shopify_sales_transform.drop(columns=["TEMP_lineitems_taxlines_rate"], inplace=True) # drop merged columns

# TAX PRICES
# recalculate tax_price from filled NaN values in tax rate
shopify_sales_transform["TEMP_lineitems_taxlines_price"] = ((shopify_sales_transform['lineitems_price'] - (shopify_sales_transform['lineitems_price'] * shopify_sales_transform['lineitems_discountallocations_amount']))
                                                            - ((shopify_sales_transform['lineitems_price'] - (shopify_sales_transform['lineitems_price'] * shopify_sales_transform['lineitems_discountallocations_amount']))
                                                               * (1/(1+shopify_sales_transform['lineitems_taxlines_rate']))))
shopify_sales_transform["lineitems_taxlines_price"].fillna(shopify_sales_transform["TEMP_lineitems_taxlines_price"], inplace=True)
shopify_sales_transform.drop(columns=["TEMP_lineitems_taxlines_price"], inplace=True) # drop merged columns

# #### 3.2.3.5 Categoricals

#SPECIAL transformations before fillna

# orders_customer_locale
# simplify to first 2 characters of string
shopify_sales_transform["orders_customer_locale"] = shopify_sales_transform["orders_customer_locale"].str[0:2]
shopify_sales_transform["orders_customer_locale"].replace("Un","Unknown",inplace=True)
shopify_sales_transform["orders_customer_locale"] = shopify_sales_transform["orders_customer_locale"].astype("category")

# FILLNA
# create dict of category columns to use fillna/replace on, replacing "" and NaN with "Unknown"
dict_cats_fillna = pd.Series(shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "category")].fillna_individual.values,index=shopify_sales_cleaning_1.loc[(shopify_sales_cleaning_1["dtype"] == "category")].feature).to_dict()
# replace "" and NaN with "Unknown" Category
for c in dict_cats_fillna:
    if "Unknown" not in list(shopify_sales_transform[c].cat.categories):
        print(c)
        shopify_sales_transform[c] = shopify_sales_transform[c].cat.add_categories('Unknown')
    shopify_sales_transform[c].replace("", "Unknown", inplace=True)
    shopify_sales_transform[c].cat.remove_unused_categories(inplace=True)
shopify_sales_transform.fillna(value=dict_cats_fillna, inplace=True)

#SPECIAL transformations after fillna

# orders_gateway
# special consolidation of "Klarna" payment methods to one
orders_gateway_consolidation_list = list(shopify_sales_transform.orders_gateway.unique()) # generate list of all unique values in orders_gateway column
# get list of all Klarna related values to replace with "Klarna"
sub = 'larna'
to_replace = [s for s in orders_gateway_consolidation_list if sub.lower() in s.lower()]
if "Klarna" in to_replace:
    to_replace.remove("Klarna")
shopify_sales_transform["orders_gateway"].replace(to_replace, "Klarna", inplace=True)
    
#CONVERSION TO VARIABLE GROUPS OF BOOLEANS

# save all new column names into a list to use in aggregation later
list_for_cat_aggregation = list()
# function turns a feature into several boolean features based on list of categoricals to transform, rest is put in "other"
def create_boolean_cat_variable_group(feature, list_of_cats, list_for_cat_agg, df):
    df[(feature+"_"+"other")] = 0
    for i in list_of_cats:
        df[(feature+"_"+i)] = (df[feature] == i)
        df[(feature+"_"+"other")] = (df[(feature+"_"+"other")] + (df[feature] == i))
        list_for_cat_agg.append((feature+"_"+i))
        df.astype({(feature+"_"+i):"boolean"})
    df[(feature+"_"+"other")] = (df[(feature+"_"+"other")] == 0)
    list_for_cat_agg.append((feature+"_"+"other"))
    return df, list_for_cat_agg
# orders_customer_locale: Only differentiate between english, spanish and german
shopify_sales_transform, list_for_cat_aggregation = create_boolean_cat_variable_group("orders_customer_locale",["en","es","de"], list_for_cat_aggregation, shopify_sales_transform)
#orders_discountcodes_type
shopify_sales_transform, list_for_cat_aggregation = create_boolean_cat_variable_group("orders_discountcodes_type",["fixed_amount","percentage"], list_for_cat_aggregation, shopify_sales_transform)
#orders_gateway
shopify_sales_transform, list_for_cat_aggregation = create_boolean_cat_variable_group("orders_gateway",["paypal","Klarna","directebanking","shopify_payments"], list_for_cat_aggregation, shopify_sales_transform)
#orders_processing_method
shopify_sales_transform, list_for_cat_aggregation = create_boolean_cat_variable_group("orders_processing_method",["free","express"], list_for_cat_aggregation, shopify_sales_transform)

# create dict of new columns to use in aggreagtion
dict_for_cat_aggregation = dict.fromkeys(list_for_cat_aggregation, "mean") # create dict with "mean" as aggregation function


# ### 3.2.4 Aggregation to day x sku level

# In[24]:


# remove last nan values
shopify_sales_transform = shopify_sales_transform[pd.isna(shopify_sales_transform["orders_created_at"]) == False]
shopify_sales_transform.isna().sum()
#shopify_sales_transform[(pd.isna(shopify_sales_transform["orders_closed_after"]) == True)&(shopify_sales_transform["lineitems_sku"] =="FP-TASSE-PANORAMA")].sort_values("orders_created_at", ascending=False).head(60)


# #### 3.2.4.1 Perform aggregation

# In[25]:


aggregation_dict = {"lineitems_quantity": "sum",
                    "lineitems_price": "mean",
                    "lineitems_gift_card": "mean",
                    "lineitems_variant_inventory_management": "mean",
                    "lineitems_discountallocations_amount": "mean",
                    "lineitems_taxlines_rate": "mean",
                    "lineitems_taxlines_price": "mean",
                    "lineitems_preorder":"mean",
                    "orders_id": "count",
                    "orders_closed_after":"mean",
                    "orders_cancelled":"mean",
                    "orders_processed_after":"mean",
                    "orders_currency_EUR": "mean",
                    "orders_presentment_currency_EUR":"mean",
                    "orders_source_web":"mean",
                    "orders_total_weight":"mean",
                    "orders_buyer_accepts_marketing":"mean", 
                    "orders_customer_total_spent":"mean",
                    "orders_customer_orders_new":"mean",
                    "orders_total_price":"mean",
                    "orders_current_total_price":"mean",
                    "orders_subtotal_price":"mean",
                    "orders_current_subtotal_price":"mean",
                    "orders_total_line_items_price":"mean",
                    "orders_total_discounts":"mean",
                    "orders_current_total_discounts":"mean",
                    "orders_shippinglinesprice":"mean",
                    "orders_shippinglinesdiscounted_price":"mean"}
# add one hot encoded to dict
aggregation_dict.update(dict_for_cat_aggregation)


# In[26]:


# create groupby variable
grouped = shopify_sales_transform.groupby([#"orders_shippingaddress_country_code",
                                           "lineitems_sku",
                                           "orders_created_at"]
                                          ,as_index=False) # maybe groupby variant_id?

# group all "to group" columns with respective aggregation functions
shopify_sales_aggregated = grouped.agg(func=aggregation_dict).copy()
shopify_sales_aggregated


# In[27]:


# substract cancelled orders
shopify_sales_aggregated["lineitems_quantity"] = (shopify_sales_aggregated["lineitems_quantity"] * (1-shopify_sales_aggregated["orders_cancelled"]))


# #### 3.2.4.2. Merge with productxdays sceleton

# In[28]:


# change dtype of shopify_productxdays_sceleton to merge
shopify_productxdays_sceleton["daydate"] = pd.to_datetime(shopify_productxdays_sceleton["daydate"], utc=True).dt.date


# In[29]:


# merge, incl. on missing group id features ("lineitems_product_id", "lineitems_sku" are now pulled in from shopify_products)
shopify_sales = shopify_productxdays_sceleton.merge(shopify_sales_aggregated, how="left", left_on=["variant_sku","daydate"], right_on=["lineitems_sku","orders_created_at"])


# ## 3.3 Clean data from transformation and aggregation #round3

# ### 3.3.1 Drop unneccessary features

# In[30]:


# nochmal kritisch hinterfragen ob nicht features gelöscht werden können? Bspw. order Preise?
shopify_sales.drop(columns=["orders_created_at"], inplace=True)


# ### 3.3.2 Clean data, filling NaNs and unwanted zeros

# In[31]:


# cleaning dictionary, first entry for fillna and second for 0 filling
# ERROR -> Throw error if NaN value or 0 value
sales_agg_cleaning = [['product_category_number', "ERROR", "SKIP"],
 ['product_category', "ERROR", 9999],
 ['product_id', "ERROR", "ERROR"],
 ['variant_sku', "ERROR", "ERROR"],
 ['variant_id', "ERROR", "ERROR"],
 ['daydate', "ERROR", "ERROR"],
 ['variant_grams', "ERROR", "SKIP"],
 ['variant_RRP', "ERROR", "SKIP"], 
 ['variant_taxable', True, "SKIP"],
 ['variant_position', 0, "SKIP"],
 ['variant_created_at', "ERROR", "ERROR"],
 ['variant_inventory_item_id', "ERROR", "ERROR"],
 ['variant_requires_shipping', True, "SKIP"],
 ['variant_inventory_management_used', 0, "SKIP"],
 ['product_status', "ERROR", "ERROR"],
 ['product_published_at', "ERROR", "ERROR"],
 ['product_published_scope', "ERROR", "ERROR"],
 ['variant_max_updated_at', "ERROR", "ERROR"],
 ['lineitems_sku', "variant_sku", "variant_sku"], # OTHER COLUMN FILL
 #['orders_created_at', 0, 0], #DELETED
 ['lineitems_quantity', 0, "SKIP"],
 ['lineitems_price', "variant_RRP", "SKIP"], # OTHER COLUMN FILL
 ['lineitems_gift_card', 0, "SKIP"],
 ['lineitems_variant_inventory_management', "variant_inventory_management_used", "SKIP"], # OTHER COLUMN FILL
 ['lineitems_discountallocations_amount', 0, "SKIP"],
 ['lineitems_taxlines_rate', 0, "SKIP"],
 ['lineitems_taxlines_price', 0, "SKIP"],
 ['lineitems_preorder', 0, "SKIP"],
 ['orders_id', 0, "SKIP"],
 ['orders_closed_after', 0, "SKIP"],
 ['orders_cancelled', 0, "SKIP"],
 ['orders_processed_after', 0, "SKIP"],
 ['orders_currency_EUR', 1, "SKIP"],
 ['orders_presentment_currency_EUR', 1, "SKIP"],
 ['orders_source_web', 1, "SKIP"],
 ['orders_total_weight', 0, "SKIP"],
 ['orders_buyer_accepts_marketing', 0, "SKIP"],
 ['orders_customer_total_spent', 0, "SKIP"],
 ['orders_customer_orders_new', 0, "SKIP"],
 ['orders_total_price', 0, "SKIP"],
 ['orders_current_total_price', 0, "SKIP"],
 ['orders_subtotal_price', 0, "SKIP"],
 ['orders_current_subtotal_price', 0, "SKIP"],
 ['orders_total_line_items_price', 0, "SKIP"],
 ['orders_total_discounts', 0, "SKIP"],
 ['orders_current_total_discounts', 0, "SKIP"],
 ['orders_shippinglinesprice', 0, "SKIP"],
 ['orders_shippinglinesdiscounted_price', 0, "SKIP"],
 ['orders_customer_locale_en', 0, "SKIP"],
 ['orders_customer_locale_es', 0, "SKIP"],
 ['orders_customer_locale_de', 0, "SKIP"],
 ['orders_customer_locale_other', 0, "SKIP"],
 ['orders_discountcodes_type_fixed_amount', 0, "SKIP"],
 ['orders_discountcodes_type_percentage', 0, "SKIP"],
 ['orders_discountcodes_type_other', 0, "SKIP"],
 ['orders_gateway_paypal', 0, "SKIP"],
 ['orders_gateway_Klarna', 0, "SKIP"],
 ['orders_gateway_directebanking', 0, "SKIP"],
 ['orders_gateway_shopify_payments', 0, "SKIP"],
 ['orders_gateway_other', 0, "SKIP"],
 ['orders_processing_method_free', 0, "SKIP"],
 ['orders_processing_method_express', 0, "SKIP"],
 ['orders_processing_method_other', 0, "SKIP"]]
sales_agg_cleaning = pd.DataFrame(sales_agg_cleaning, columns=["feature","fillna","zeros"])
sales_agg_cleaning


# In[32]:


def after_finalmerge_cleaning(df, cleaning_instructions):
    for feature in list(cleaning_instructions.feature):
        current_row = cleaning_instructions[cleaning_instructions["feature"] == feature]
        all_features = list(df.columns)
        
        # check if raise ERROR because of NaN values:
        if (df[feature].isnull().sum() > 0) & ((current_row["fillna"].iloc[0]) == "ERROR"):
            raise NameError(feature, " still has not acceptable NaN values.")
            
        # check if raise ERROR because of null values:
        if (((df[feature]==0).sum()) > 0) & ((current_row["zeros"].iloc[0]) == "ERROR"):
            raise NameError(feature, " still has not acceptable zero (0) values.")
            
        """ from here NaN values """
        # other column fills first
        if current_row["fillna"].iloc[0] in all_features:
            df[feature] = np.where((pd.isna(df[feature]) == True), df[(current_row["fillna"].iloc[0])], df[feature])
        
        # fill with values from cleaning_instructions
        if ((current_row["fillna"].iloc[0]) != "SKIP")&((current_row["fillna"].iloc[0]) != "ERROR"):
            df[feature].fillna((current_row["fillna"].iloc[0]), inplace=True)
        
        """ from here zero (0) values """
        # other column fills first
        if current_row["zeros"].iloc[0] in all_features:
            df[feature] = np.where(df[feature] == 0, df[(current_row["zeros"].iloc[0])], df[feature])
            
        # fill with values from cleaning_instructions
        if ((current_row["zeros"].iloc[0]) != "SKIP")&((current_row["zeros"].iloc[0]) != "ERROR"):
            df.loc[df[feature] == 0, feature] = (current_row["zeros"].iloc[0])


# In[33]:


after_finalmerge_cleaning(shopify_sales, sales_agg_cleaning) # cleaning
shopify_sales.fillna(0, inplace=True) # final fillna with 0
shopify_sales


# ## 3.4 Feature engineering

# ### Features to engineer:
# - seasonality and trend break down
# - log of target (lineitems_quantity)
# - rolling monthly/weekly averages and differences -> Give last years as future data input
# 
# USE LATER:
# - EXTERNAL: include shopify EUR to USD converation rate?
# - INPUT: number of available payment methods
# - INPUT: shipping-fee threshold
# - Number of different shipping methods and fees available
# - outstanding giftcards --> Keep track of outstanding number and value of giftcards
# - sellout quantity with and without discount

# In[34]:


def lag_by_oneYear(df, column_name_to_lag, identifier):
    df = df.copy()
    df_newData = df.copy()
    df_newData["daydate_plus1y"] = df_newData["daydate"] + relativedelta(months = 12)
    df_newData = df_newData[[identifier,"daydate_plus1y",column_name_to_lag]]
    df_newData = df_newData.groupby(["daydate_plus1y",identifier], as_index=False).mean() # to eliminate duplicates in leap years 
                                                            # (e.g. 2020-02-29 -> will become 2021-02-28 so we have that double in 2021)
    df_newData.rename(columns={column_name_to_lag:column_name_to_lag+"_lastYear"}, inplace=True)
    #df.drop(columns="daydate_plus1y", inplace=True)
    df = df.merge(df_newData, how="left", left_on=[identifier, "daydate"], right_on=[identifier, "daydate_plus1y"])
    df.drop(columns="daydate_plus1y", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=[identifier,"daydate"], inplace=True)
    df[column_name_to_lag+"_lastYear"] = df[column_name_to_lag+"_lastYear"].fillna(method="ffill") # to fill nans 
                                                            # coming from 02-29 as they have no value from previous year
    return df


# ### 3.4.1 Rolling averages and differences

# In[35]:


# sort dataframe the other way around
shopify_sales = shopify_sales.sort_values(["variant_sku", "daydate"],ascending=[True,True]).reset_index(drop=True)
shopify_sales


# In[36]:


# specify which features to transform
features_to_avg = ["lineitems_quantity"]


# In[37]:


# rolling average last 7 days
for i in features_to_avg:
    shopify_sales["rolling7days_"+i] = shopify_sales.groupby(by=["variant_id"])[i].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
# rolling average last 30 days
for i in features_to_avg:
    shopify_sales["rolling30days_"+i] = shopify_sales.groupby(by=["variant_id"])[i].transform(lambda x: x.rolling(30, min_periods=1).mean())


# In[38]:


# difference to rolling
for i in features_to_avg:
    shopify_sales["delta7days_"+i] = shopify_sales[i] - shopify_sales["rolling7days_"+i]
    shopify_sales["delta30days_"+i] = shopify_sales[i] - shopify_sales["rolling30days_"+i]


# In[39]:


# Fill last years 30 days rolling averages in this years

for i in features_to_avg:
    shopify_sales = lag_by_oneYear(shopify_sales, "rolling7days_"+i, "variant_id")
    
shopify_sales.fillna(0, inplace=True)


# In[40]:


# Fill last years 7 days rolling averages in this years

for i in features_to_avg:
    shopify_sales = lag_by_oneYear(shopify_sales, "rolling30days_"+i, "variant_id")
    
shopify_sales.fillna(0, inplace=True)


# ### 3.4.2 Logs

# In[41]:


# log
for i in features_to_avg:
    shopify_sales["log_"+i] = np.log(shopify_sales[i] + 1e-8)


# ### 3.4.3 STL decomposition

# In[42]:


def STL_perProduct(df, identifier, feature_toSTL):
    STL_results = pd.DataFrame(columns=["variant_id","daydate"])

    for p in list(df[identifier].unique()):
        res = STL(df[df[identifier] == p][["daydate",feature_toSTL]].set_index("daydate"), period=30).fit()
        plevel_results = pd.DataFrame(res.trend, columns=["trend"])
        plevel_results = pd.concat([plevel_results, pd.DataFrame(res.seasonal, columns=["season"])], axis=1)
        plevel_results = pd.concat([plevel_results, pd.DataFrame(res.resid, columns=["resid"])], axis=1)
        plevel_results.reset_index(inplace=True)
        plevel_results["variant_id"] = p
        plevel_results.rename(columns={"trend":feature_toSTL+"_trend","season":feature_toSTL+"_season","resid":feature_toSTL+"_resid"}, inplace=True)
        STL_results = STL_results.append(plevel_results)
        
    df = df.merge(STL_results, how="left", on=[identifier,"daydate"])
    df.drop_duplicates(inplace=True)
    
    return df


# In[43]:


# do it for 7 days rolling, as it will smoothen a little more
shopify_sales = shopify_sales.sort_values(by=["variant_id","daydate"])
shopify_sales = STL_perProduct(shopify_sales, "variant_id", "rolling7days_lineitems_quantity")
shopify_sales.fillna(0, inplace=True)


# In[44]:


# fill +1 year with previous year trend, season and resid
shopify_sales = shopify_sales.sort_values(by=["variant_id","daydate"])
for i in ["trend","season","resid"]:
    shopify_sales = lag_by_oneYear(shopify_sales, ("rolling7days_lineitems_quantity_"+i), "variant_id")
    shopify_sales.fillna(0, inplace=True)


# In[45]:


shopify_sales.reset_index(drop=True, inplace=True)


# # 4. Finalization

# In[46]:


#shopify_sales


# In[47]:


# change False / True
shopify_sales["lineitems_variant_inventory_management"] = shopify_sales["lineitems_variant_inventory_management"].apply(lambda x: 1 if x == True else 0)


# # 5. Read to database


# In[49]:


t = Timer("Export")


# In[50]:


shopify_sales.head(5).to_sql('shopify_sales', con = engine, schema="transformed", if_exists='replace',index=False) #drops old table and creates new empty table
shopify_sales.head(5).to_sql('historical_sales', con = engine, schema="sc_shopify", if_exists='replace',index=False) #drops old table and creates new empty table

# In[51]:


engine.execute('TRUNCATE TABLE transformed.shopify_sales') #Truncate the table in case you've already run the script before
engine.execute('TRUNCATE TABLE sc_shopify.historical_sales') #Truncate the table in case you've already run the script before

# In[52]:


engine.dispose()


# In[53]:


shopify_sales.to_csv('/opt/ml/processing/input/Data Source Transformations/Data Uploads/shopify_sales.csv', index=False, header=False)


# In[54]:


col_names = ", ".join(map(lambda x: '"'+x+'"', list(shopify_sales.columns)))


# In[55]:


with open('/opt/ml/processing/input/Data Source Transformations/Data Uploads/shopify_sales.csv', 'r') as f:    
    conn = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database")).raw_connection()
    cursor = conn.cursor()
    cmd = 'COPY transformed.shopify_sales('+ col_names+ ')'+ 'FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
    cursor.copy_expert(cmd, f)
    conn.commit()
    
engine.dispose()

with open('/opt/ml/processing/input/Data Source Transformations/Data Uploads/shopify_sales.csv', 'r') as f:    
    conn = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database")).raw_connection()
    cursor = conn.cursor()
    cmd = 'COPY sc_shopify.historical_sales('+ col_names+ ')'+ 'FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
    cursor.copy_expert(cmd, f)
    conn.commit()

# In[56]:


t.end()


# engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"),echo=False) # do not use use_batch_mode=True

# t = Timer("Export")
# shopify_sales.to_sql("shopify_sales", con = engine, schema="transformed", if_exists='replace', index=False, chunksize=1000, method="multi")
# t.end()

# In[57]:


engine.dispose()


# In[58]:


print(params)

