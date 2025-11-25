import pandas as pd
import numpy as np
from parameter_store_client import EC2ParameterStore
from Support_Functions import *
from sqlalchemy import create_engine
import logging
import hashlib
import os
import sys
from google.oauth2 import service_account
import pandas_gbq


# define system variables for sagemaker environment
def set_system_variables():
    prefix = '/opt/ml'
    src  = os.path.join(prefix, 'processing/input/')
    sys.path.append(src)

def fetch_parameters_from_store(dbtype="dev") -> dict:
    """
    Fetches parameters from AWS Parameter Store
    """
    try:
        store = EC2ParameterStore()
        if dbtype=="dev":
            parameters = store.get_parameters_by_path("/development/db")
        else:
            parameters = store.get_parameters_by_path("/production/db/")
        return parameters
    except Exception as e:
        logging.error(f"Error occurred while fetching parameters from store: {e}")
        raise

def get_channel_name(client_name):
    """
    Fetches the parameters from AWS to get channel names
    """
    try:
        store = EC2ParameterStore()
        parameters = store.get_parameters_by_path("/"+client_name+"/")
        # channel names splitted via single comma 
        # TODO : return as a list
        return parameters.get("sales_channel").split(",")
    except Exception as e:
        logging.error(f"Error occurred while fetching parameters from store: {e}")
        raise


def get_db_connection(dbtype="dev", client_name:str=None) -> str:
    """
    Returns the database connection string
    """
    try:
        try:
            parameters = fetch_parameters_from_store(dbtype=dbtype)
        except Exception as e:
            logging.error(f"Error occurred while fetching parameters from store: {e}")
            raise

        db_user = parameters.get("POSTGRES_USERNAME")
        db_password = parameters.get("POSTGRES_PASSWORD")
        db_host = parameters.get("POSTGRES_DB")
        db_port = parameters.get("POSTGRES_PORT")
        db_name = client_name

        # TODO : development parameters need to be fixed in parameters store 
        if dbtype=="dev":
            db_host = os.getenv("DEV_DB_HOST", "nogl-dev.c2wwnfcaisej.eu-central-1.rds.amazonaws.com")
            db_password = os.getenv("DEV_DB_PASSWORD", "")
            if not db_password:
                raise ValueError(
                    "Missing DEV_DB_PASSWORD environment variable for development database. "
                    "Please set it in your .env file or environment."
                )

        engine = create_engine('postgresql://'+db_user+":"+db_password+"@"+db_host+":"+db_port+"/"+db_name,echo=False)
        return engine
    
    except Exception as e:
        logging.error(f"Error occurred while creating an engine: {e}")
        raise

def import_data_AWSRDS(table, schema, engine):
    chunks = list()
    for chunk in pd.read_sql("SELECT * FROM "+schema+"."+table, con=engine, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(list(chunks))
    return df


def find_unique_columns(data_frame):
    unique_columns = []
    
    for column in data_frame.columns:
        if data_frame[column].nunique() == len(data_frame):
            unique_columns.append(column)
    return unique_columns


def get_identifier_keys(channel_name):
    identifier_keys = {
        "amazon": ["variant_sku",
              "product_id",
              "product_category",
              "product_title", 
              "variant_title", 
              "listing_id",
              "variant_barcode",
              "product_category_number",
              "product_status",
              "variant_RRP",
              "variant_asin"],

        "shopify": ["variant_sku",
               "variant_id",
               "product_id", 
               "product_title", 
               "variant_title",
               "variant_RRP",
               "product_category",
               "product_category_number",
               "product_status",
               "variant_inventory_item_id",
               "variant_barcode" ],
    }
    return identifier_keys[channel_name]

def clean_data(df, channel):
    """"
    Cleans the data for each channel
    """
    if channel == "amazon":
        # First  change dtype of None values of NaN
        df["variant_barcode"] = df["variant_barcode"].replace("None", np.nan)
        df["variant_barcode"] = df["variant_barcode"].fillna(df["product_id"])
        df = df[["variant_sku", "variant_RRP", "variant_barcode", "product_status", "variant_asin", "product_category"]]
        
    elif channel == "shopify":
        # replace None values of variant barcode with variant_id since assumption is hold.for shopify
        df["variant_barcode"] = df["variant_barcode"].replace("", np.nan)
        df["variant_barcode"] = df["variant_barcode"].fillna(df["variant_id"])
        df = df[["variant_sku", "variant_RRP", "variant_barcode", "product_status", "variant_id", "product_category"]]

    return df

def rename_columns_with_prefix(df, prefix):
    for c in df.columns:
        df.rename(columns={c:prefix+c}, inplace=True)
    return df


def find_duplicate_variants(df, grouping_ids):
    """
    Finds rows in the DataFrame where the variant_barcode and product_status are the same, 
    but the variant_sku is different.
    
    Parameters:
    - df: The input DataFrame.
    
    Returns:
    - DataFrame containing only the rows with the described duplicates.
    """
    # Group by variant_barcode and product_status
    groups = df.groupby(grouping_ids)
    duplicate_rows = []
    for _, group in groups:
        if group['variant_sku'].nunique() > 1:
            duplicate_rows.append(group)
        
    # If there are no duplicates, return an empty dataframe with the same columns
    if not duplicate_rows:
        return pd.DataFrame(columns=df.columns)

    # Concatenate the groups with duplicates and return
    return pd.concat(duplicate_rows, axis=0)


# Mark duplicates using the provided function
def mark_duplicates(df):
    # Check based on barcode and product status
    barcode_duplicates = find_duplicate_variants(df, ['variant_barcode', 'product_status'])
    print("barcode level duplictes:", len(barcode_duplicates))
    # Check based on SKU and product status
    sku_duplicates = find_duplicate_variants(df, ['variant_sku', 'product_status'])
    print("sku level duplictes:", len(sku_duplicates))
    # Combine and return the marked duplicates
    return pd.concat([barcode_duplicates, sku_duplicates], axis=0).drop_duplicates()


def hash_row(row):
    """Generate a unique hash for a dataframe row."""
    hash_hex = hashlib.md5(str(row.values).encode()).hexdigest()
    short_hash = hash_hex[:16]
    return short_hash


def handle_channel_level_duplicates(df):
    # a)duplicates with the same variant_barcode and product_status but different variant_sku in one sales channel 
    # -> treat as individual variant, generate individual forecasts and notify the customer about the duplicates
    sku_df = df.groupby(["variant_sku", "product_status"]).agg({"variant_barcode": lambda x: list(x)}).reset_index()
    # if the variant_sku is same then variant_barcode should be same
    sku_df = sku_df[sku_df["variant_barcode"].apply(lambda x: len(set(x)))>1]

    if len(sku_df) == 0:
        print("no duplicates at sku level")
        # just add new column to avoid error
        df["local_nogl_id"] = ""

    else:
        print("sku level duplictes:", len(sku_df))
        # if variant_sku column value length in barcode_df is gretaer than 1 we treat those as different products so we should assign different unique id to those entries
        # if variant_sku column value length in barcode_df is 1 we treat those as same products so we should assign same unique id to those entries
        sku_df['barcode_count'] = sku_df['variant_barcode'].apply(len)
        for i in range(len(sku_df)):
            if sku_df['barcode_count'][i] == 1:
                # assign same unique id to all variant_barcodes in df
                # df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][0], 'local_nogl_id'] = df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][0]].apply(hash_row, axis=1)
                df.loc[df['variant_barcode'] == sku_df['variant_barcode'][i][0], 'local_nogl_id'] = df.loc[df['variant_barcode'] == sku_df['variant_barcode'][i][0]].apply(hash_row, axis=1)
            else:
                # assign different unique id to all variant_barcodes in df
                for j in range(len(sku_df['variant_barcode'][i])):
                     df.loc[df['variant_barcode'] == sku_df['variant_barcode'][i][j], 'local_nogl_id'] = df.loc[df['variant_barcode'] == sku_df['variant_barcode'][i][j]].apply(hash_row, axis=1)
                
                
    # if variant_sku is the same then assign same unique id to those entries in df
    #b) duplicates with the same variant_sku and product_status but different variant_barcode 
    # -> treat as individual variant, generate individual forecasts and notify the customer about the duplicates
    barcode_df = df.groupby(["variant_barcode", "product_status"]).agg({"variant_sku": lambda x: list(x)}).reset_index()
    # if the variant_sku is same then variant_barcode should be same
    barcode_df = barcode_df[barcode_df["variant_sku"].apply(lambda x: len(set(x)))>1]
    # if variant_sku is the same then assign same unique id to those entries in df
    #barcode_df['sku_count'] = barcode_df['variant_sku'].apply(len)
    if len(barcode_df)  == 0:
        print("no duplicates at barcode level")
        df["local_nogl_id"] = ""
    else:
        # if variant_barcode column value length in sku_df is gretaer than 1 we treat those as different products so we should assign different unique id to those entries
        # if variant_barcode column value length in sku_df is 1 we treat those as same products so we should assign same unique id to those entries
        print("barcode level duplictes:", len(barcode_df))
        barcode_df['sku_count'] = barcode_df['variant_sku'].apply(len)
        for i in range(len(barcode_df)):
            if barcode_df['sku_count'][i] == 1:
                # assign same unique id to all variant_skus in df
                # final_products["nogl_id"] = final_products.apply(hash_row, axis=1)
                df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][0], 'local_nogl_id'] = df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][0]].apply(hash_row, axis=1)
                #df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][0], 'unique_id'] = hash_row(barcode_df.iloc[i])
            else:
                # assign different unique id to all variant_skus in df
                for j in range(len(barcode_df['variant_sku'][i])):
                    df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][j], 'local_nogl_id'] = df.loc[df['variant_sku'] == barcode_df['variant_sku'][i][j]].apply(hash_row, axis=1)

    return df, sku_df, barcode_df


def get_fused_variants(df, id_col):
    # return grouped df by unique_id
    df = df.groupby(id_col).agg({"variant_sku": lambda x: list(x), "variant_barcode": lambda x: list(x), "product_status": lambda x: list(x)}).reset_index()
    df["variant_sku"] = df["variant_sku"].apply(lambda x: x[0])
    df["variant_barcode"] = df["variant_barcode"].apply(lambda x: x[0])
    df["product_status"] = df["product_status"].apply(lambda x: x[0])
    return df



def concat_duplicates_in_channel_level(df_duplicates, df_no_duplicates):
    # df_duplicates : we already handled the duplicates (e.g. products fused based on the conditions)
    # Now we can concat together again after resolved the duplicates and assign local nogl id to others. 
    # But lets check one more time after concat if there are still duplicates
    # add prefix the columns first
    
    check_df = find_duplicate_variants(df_no_duplicates, ['variant_barcode', 'product_status'])
    if len(check_df) > 0:
        raise Exception("There are still duplicates in df_no_duplicates!!!!")
    else:
        # assign local nogl id to df_no_duplicates
        df_no_duplicates["local_nogl_id"] = df_no_duplicates.apply(hash_row, axis=1)


    concated_df = pd.concat([df_duplicates, df_no_duplicates], axis=0)
    concated_df = concated_df.reset_index(drop=True)
    # drop duplicates
    concated_df = concated_df.drop_duplicates(subset=["variant_sku", "variant_barcode", "product_status"], keep="first")
    # check if there are still duplicates
    duplicates = concated_df[concated_df.duplicated(subset=["variant_sku", "variant_barcode", "product_status"], keep=False)]
    print("duplicates after concat:", len(duplicates))
    if len(duplicates) > 0:
        raise Exception("There are still duplicates after concat")
    
    
    df_duplicates = rename_columns_with_prefix(df_duplicates, prefix="duplicates_")
    df_no_duplicates = rename_columns_with_prefix(df_no_duplicates, prefix="no_duplicates_")
    # reformat the values in the columns of duplicates df


    merge_variants = concated_df.merge(df_duplicates,
        left_on = ["variant_sku", "variant_barcode"], 
        right_on = ["duplicates_" + "variant_sku",  "duplicates_" + "variant_barcode"], 
        how="left")

    merge_variants = merge_variants.merge(df_no_duplicates, 
        left_on = ["variant_sku", "variant_barcode"], 
        right_on = ["no_duplicates_" + "variant_sku", "no_duplicates_" + "variant_barcode"], 
        how="left")

    
    #concated_df = concated_df.reset_index(drop=True)
    #return concated_df, merge_variants

    return merge_variants

# return rows that does not contain any NaN values
def find_rows_without_nan(df):
    # return rows that does not contain any NaN values
    return df[~df.isna().any(axis=1)]



def add_field(source_df, target_df, channel, new_field):
    # TODO : make search based on the columns of target_df that contains channel name on source_df to find exact variant and get its field value
    cols = [col for col in target_df.columns if channel in col]
    target_df = target_df[cols]
    target_df = target_df.rename(columns={col:col.replace(channel+"_", "") for col in cols})
    cols = [col.replace(channel+"_", "") for col in cols]
    cols_source = cols
    cols_source.append(new_field)
    print(source_df.columns)
    print(cols)
    source_df = source_df[cols_source]
    # remove new field from cols
    cols.remove(new_field)

    merged_df = source_df.merge(target_df, left_on=cols, right_on=cols, how="left")
    # add field column to source_df
    # rename the new_field column add channel name as prefix
    merged_df = merged_df.rename(columns={new_field:channel+"_"+new_field})
    return merged_df
    

def get_value_or_fill(amazon_value, shopify_value):
    if pd.isna(amazon_value) or amazon_value == 'nan':
        return shopify_value
    elif pd.isna(shopify_value) or shopify_value == 'nan':
        return amazon_value
    elif amazon_value == shopify_value:
        return amazon_value
    else:
        return None


# add nogl_id to sceletons
# TODO : Fort shopify_sceleton :  try to match shopify_variant_id , shopify_variant_sku and shopify_product_status to get nogl_id
# TODO : For Amazon sceleton : try to match amazon_variant_asin, amazon_variant_sku and amazon_product_status to get nogl_id
def add_nogl_id_to_sceleton(id_df, sceleton_df, sales_channel):
    if sales_channel == "shopify":
        print("shopify")
        # Rename the columns for merging (remove sales_channel prefix)
        id_df = id_df.rename(columns={
            "shopify_variant_id": "variant_id", 
            "shopify_variant_sku": "variant_sku", 
            "shopify_variant_RRP" : "variant_RRP",
            "shopify_product_status": "product_status",
            "shopify_variant_barcode" : "variant_barcode"
        })

        print(id_df.columns)

        # Drop the columns with original shopify prefixes to avoid duplicates
        #columns_to_drop = ['shopify_variant_id', 'shopify_variant_sku', 'shopify_variant_barcode', 'shopify_product_status']
        #id_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Perform the merge
        sceleton_df = sceleton_df.merge(id_df[["nogl_id", "variant_id", "variant_sku", "variant_barcode", "product_status"]], 
                                        how="left", 
                                        on=["variant_id", "variant_sku", "product_status"])
    
    elif sales_channel == "amazon":
        print("amazon")
        # rename the columns for merging (remove sales_channel prefix)
        id_df = id_df.rename(columns={
            "amazon_variant_asin": "variant_asin", 
            "amazon_variant_sku": "variant_sku",
            "amazon_variant_RRP": "variant_RRP", 
            "amazon_product_status": "product_status"
        })
        
        # also rename amazon_variant_barcode to product_id
        id_df = id_df.rename(columns={"amazon_variant_barcode": "product_id"})
        
        # Merge based on multiple columns
        sceleton_df = sceleton_df.merge(id_df[["nogl_id", "product_id", "variant_asin", "variant_sku", "product_status"]], 
                                    how="left", 
                                    on=["product_id", "variant_asin", "variant_sku", "product_status"])

        # Check if any rows are missing after the merge
        missing_rows = sceleton_df["nogl_id"].isnull()

        if missing_rows.sum() > 0:
            # extract those missing rows into a df
            missing_rows_df = sceleton_df[missing_rows]
            # now try to fill those missing rows with the help of variant_sku and variant_asin
            missing_rows_df = missing_rows_df.merge(id_df[["nogl_id", "variant_asin", "variant_sku", "product_status"]],
                                                    how="left",
                                                    on=["variant_asin", "variant_sku"])
            # print duplicates in missing_rows_df
            # merge missing_rows_df with sceleton_df to add nogl_id_y to sceleton_df
            sceleton_df = sceleton_df.merge(missing_rows_df[["nogl_id_y", "daydate", "variant_asin", "variant_sku"]],
                                            how="left",
                                            on=["daydate", "variant_asin", "variant_sku"])
            # renamce nogl_id to nogl_id_x 
            sceleton_df = sceleton_df.rename(columns={"nogl_id": "nogl_id_x"})
            # create new column nogl_id and fill with nogl_id_x if it is not null otherwise fill with nogl_id_y
            sceleton_df["nogl_id"] = sceleton_df["nogl_id_x"].fillna(sceleton_df["nogl_id_y"])
            # drop nogl_id_y and nogl_id_x
            sceleton_df = sceleton_df.drop(columns=["nogl_id_y", "nogl_id_x"])



        # Check if any rows are missing after the merge
        missing_rows = sceleton_df["nogl_id"].isnull()
        if missing_rows.sum() > 0:
            # extract those missing rows into a df
            missing_rows_df = sceleton_df[missing_rows]
            # now try to fill those missing rows with the help of variant_sku only
            missing_rows_df = missing_rows_df.merge(id_df[["nogl_id", "variant_sku", "product_status"]],
                                                    how="left",
                                                    on=["variant_sku"])
            # print duplicates in missing_rows_df
            # merge missing_rows_df with sceleton_df to add nogl_id_y to sceleton_df
            sceleton_df = sceleton_df.merge(missing_rows_df[["nogl_id_y", "daydate", "variant_asin", "variant_sku"]],
                                            how="left",
                                            on=["daydate", "variant_asin", "variant_sku"])
            # renamce nogl_id to nogl_id_x 
            sceleton_df = sceleton_df.rename(columns={"nogl_id": "nogl_id_x"})
            # create new column nogl_id and fill with nogl_id_x if it is not null otherwise fill with nogl_id_y
            sceleton_df["nogl_id"] = sceleton_df["nogl_id_x"].fillna(sceleton_df["nogl_id_y"])
            # drop nogl_id_y and nogl_id_x
            sceleton_df = sceleton_df.drop(columns=["nogl_id_y", "nogl_id_x"])
        
        # Check if any rows are missing after the merge
        missing_rows = sceleton_df["nogl_id"].isnull()

        if missing_rows.sum() > 0:
            # extract those missing rows into a df
            missing_rows_df = sceleton_df[missing_rows]
            # now try to fill those missing rows with the help of variant_asin only
            missing_rows_df = missing_rows_df.merge(id_df[["nogl_id", "variant_asin", "product_status"]],
                                                    how="left",
                                                    on=["variant_asin"])
            #now insert those missing rows back into the sceleton_df
            # print duplicates in missing_rows_df
            # merge missing_rows_df with sceleton_df to add nogl_id_y to sceleton_df
            sceleton_df = sceleton_df.merge(missing_rows_df[["nogl_id_y", "daydate", "variant_asin", "variant_sku"]],
                                            how="left",
                                            on=["daydate", "variant_asin", "variant_sku"])
            # renamce nogl_id to nogl_id_x 
            sceleton_df = sceleton_df.rename(columns={"nogl_id": "nogl_id_x"})
            # create new column nogl_id and fill with nogl_id_x if it is not null otherwise fill with nogl_id_y
            sceleton_df["nogl_id"] = sceleton_df["nogl_id_x"].fillna(sceleton_df["nogl_id_y"])
            # drop nogl_id_y and nogl_id_x
            sceleton_df = sceleton_df.drop(columns=["nogl_id_y", "nogl_id_x"])

        # Check if any rows are missing after the merge
        missing_rows = sceleton_df["nogl_id"].isnull()
        if missing_rows.sum() > 0:
            raise Exception("There are still missing rows please check it")

    return sceleton_df


def clean_source_data(source_df, source_data_id_column, sales_channel):
    """
    Clean the source dataframe by removing sales_channel name from id if exists.
    
    :param source_df: DataFrame containing the source data
    :param source_data_id_column: Column name where the IDs with sales channel names might exist
    :return: Cleaned DataFrame
    """

    # TODO : remove sales_channel name from id if exists
    # List of possible sales channels (this is just a sample, adjust to your needs) 

    # Check if any of the sales channel name is a substring of the ID. If yes, remove it.
    for channel in sales_channel:
        mask = source_df[source_data_id_column].str.contains(channel, case=False, na=False)
        source_df.loc[mask, source_data_id_column] = source_df.loc[mask, source_data_id_column].str.replace(channel + '_', '', regex=False)
    
    source_df[source_data_id_column] = source_df[source_data_id_column].str.split('_').str[-1]
    return source_df

def klaviyo_create_productxday_skeleton (events_raw_df, campaigns_raw_df, lists_df, campaigns_list_df, product_X_day_skeleton):
        '''
        Requires 5 input data frames (events_raw_df, campaigns_raw_df, lists_df, campaigns_list_df and product_X_day_skeleton) and 
        returns 1 dataframe  containing day-by-day data for each product. 
        '''
        
        # select only the correct columns from the event df
        klaviyo_events_raw = events_raw_df[['id',
                                                'uuid',
                                                'flow_id',
                                                'campaign_id',
                                                'statistic_id',
                                                'flow_message_id',
                                                'datetime',
                                                'timestamp',
                                                'event_name',
                                                'event_properties']]
        
        # Split events by type 
        # SKU events need to be handled seperately because events_properties json is not always build the same
        klaviyo_events_dict = split_event_data(klaviyo_events_raw)

        # Create SKU event dataframes
        klaviyo_events_sku_checkoutStarted = klaviyo_events_dict['checkout_started']
        klaviyo_events_sku_orderedProduct = klaviyo_events_dict['ordered_product']
        klaviyo_events_sku_cancelledOrder = klaviyo_events_dict['cancelled_order']
        klaviyo_events_sku_refundedOrder = klaviyo_events_dict['refunded_order']

        # Create non SKU event dataframe
        klaviyo_events_nonsku = klaviyo_events_dict['non_sku_events']
        
        # Preprocess SKU events
        checkoutStarted_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_checkoutStarted, "Checkout Started")
        cancelledOrder_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_cancelledOrder, "Cancelled Order")
        refundedOrder_skus_by_date = klaviyo_Extract_sku_id_from_checkout_cancelled_and_refunded(klaviyo_events_sku_refundedOrder, "Refunded Order")
        orderedProduct_skus_by_date = klaviyo_Extract_sku_id_from_orderedProduct(klaviyo_events_sku_orderedProduct, "Ordered Product")
        
        # Preprocess capaign data and non-sku events
        nonsku_event_and_campaign_by_date = preprocess_campaigns_and_nonsku_events(campaigns_raw_df, lists_df, campaigns_list_df, klaviyo_events_nonsku)
        
        # Prepare dataframes for merging
        klaviyo_productsxdays_sceleton = product_X_day_skeleton # [['variant_sku', 'daydate', 'variant_id']]  # Drop all non-index columns 
        klaviyo_productsxdays_sceleton['daydate'] = pd.to_datetime(klaviyo_productsxdays_sceleton['daydate']) # Convert to datetime    
        
        
        klaviyo_productsxdays_sceleton = klaviyo_productsxdays_sceleton.merge(nonsku_event_and_campaign_by_date, on = "daydate", how = "left")
        
        # Drop superfluous event_name columns
        checkoutStarted_skus_by_date = checkoutStarted_skus_by_date.drop("event_name", axis = 1)
        cancelledOrder_skus_by_date = cancelledOrder_skus_by_date.drop("event_name", axis = 1)
        refundedOrder_skus_by_date = refundedOrder_skus_by_date.drop("event_name", axis = 1)
        orderedProduct_skus_by_date = orderedProduct_skus_by_date.drop("event_name", axis = 1)
        
        skuXdate_df = checkoutStarted_skus_by_date.merge(cancelledOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
        skuXdate_df = skuXdate_df.merge(refundedOrder_skus_by_date, on =['sku_id', 'datetime'], how='outer')
        skuXdate_df = skuXdate_df.merge(orderedProduct_skus_by_date, on =['sku_id', 'datetime'], how='outer')
        
        # prepare columns for merging
        skuXdate_df.rename(columns = {'datetime' : 'daydate', 'sku_id' : "variant_sku"}, inplace = True) 
        skuXdate_df['daydate'] = pd.to_datetime(skuXdate_df['daydate'])
        print(skuXdate_df.columns)
        
        # Clean event data
        skuXdate_df = skuXdate_df.fillna(0)
        skuXdate_df = skuXdate_df.drop_duplicates()
        return skuXdate_df
    


def load_transformation_data(transformation, prod_engine, dev_engine, sales_channel):
    if transformation == "google_ads":
        df_cp_mapping = load_campaignperformance(incl_name=True, prod_engine=prod_engine)
        df_pp = load_productperformance(prod_engine=prod_engine)
        source_df = clean_source_data(df_pp, "segments.product_item_id", sales_channel=sales_channel)
        return source_df

    elif transformation == "klaviyo":
        klaviyo_events_raw = import_data_AWSRDS(schema="klaviyo",table="events_scd",engine=prod_engine)
        klaviyo_campaigns_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns",engine=prod_engine)
        klaviyo_campaigns_lists_raw = import_data_AWSRDS(schema="klaviyo",table="campaigns_lists",engine=prod_engine)
        klaviyo_lists_raw = import_data_AWSRDS(schema="klaviyo",table="lists",engine=prod_engine)
        # Load Product ID Data
        klaviyo_productsxdays_sceleton = import_data_AWSRDS(schema="product",table="productsxdays_sceleton",engine=dev_engine)
        klaviyo_events_raw["event_properties"]= klaviyo_events_raw["event_properties"].astype("string")
        # Create the current klaviyo from the airbyte datasets 
        filled_productsxdays_sceleton = klaviyo_create_productxday_skeleton(klaviyo_events_raw, klaviyo_campaigns_raw, klaviyo_lists_raw, klaviyo_campaigns_lists_raw, klaviyo_productsxdays_sceleton)
        return filled_productsxdays_sceleton
    

    elif transformation == "google_analytics":
        df = load_pdp(engine=prod_engine)
        df = run_pdp_pipeline(df)
        df = fill_nan(df)
        return df

    elif transformation == "amazon_sales":
        amazon_sales = import_data_AWSRDS(schema="amazon",table="order_line_items",engine=dev_engine)
        print(amazon_sales.shape)
        amazon_sceleton = import_data_AWSRDS(schema="transformed",table="amazon_productsxdays_sceleton",engine=dev_engine)
        print(amazon_sceleton.shape)
        columns_to_keep, hierarchy, dates, keep_as_is, feature_engineering, boolean = get_columns_to_keep()
        # get rid of duplicates
        print("Before duplicates removal: ", amazon_sales.shape)
        amazon_sales.sort_values(by=["order_last_update_date", "lineitem_quantity_shipped"], ascending=[False, False], inplace=True)
        amazon_sales = amazon_sales.drop_duplicates(subset=(hierarchy + dates + ["order_amazon_order_id"]), keep='first')
        print("After duplicates removal: ", amazon_sales.shape)
        amazon_sales_clean = amazon_sales[columns_to_keep].copy()
        amazon_sales_clean = replace_none_with_nan(amazon_sales_clean)
        amazon_sales_clean = replace_nan_with_zeros(amazon_sales_clean, keep_as_is)
        # transform before aggregation
        # feature engineering
        amazon_sales_clean = compute_average_delivery_time(amazon_sales_clean)

        # date
        for c in dates:
            amazon_sales_clean[c] = pd.to_datetime(amazon_sales_clean[c], utc=True, errors='coerce').dt.strftime("%Y-%m-%d")

        # boolean
        amazon_sales_clean = make_true(amazon_sales_clean, "lineitem_is_gift", "true") # make "true" True the rest true
        amazon_sales_clean = make_true(amazon_sales_clean, "lineitem_condition_id", "New") # make "New" True the rest false
        amazon_sales_clean = make_true(amazon_sales_clean, "order_fulfillment_channel", "AFN") # make AFN True the rest false
        amazon_sales_clean = make_true(amazon_sales_clean, "order_shipment_service_level_category", "Expedited") # make Expedited True the rest false
        amazon_sales_clean = make_true(amazon_sales_clean, "order_order_total_currency_code", "EUR") # make EUR True the rest false
        amazon_sales_clean = make_true(amazon_sales_clean, "order_shipping_address_country_code", "DE") # make DE True the rest false
        amazon_sales_clean["lineitem_promotion_ids"] = amazon_sales_clean["lineitem_promotion_ids"].replace("None", np.nan)
        amazon_sales_clean = transform_to_boolean(amazon_sales_clean, boolean)
        amazon_sales_clean.rename(columns={"lineitem_promotion_ids": "lineitem_has_promotion_ids",
                                        "lineitem_condition_id": "lineitem_condition_is_new",
                                        "order_fulfillment_channel": "fullfillment_channel_is_AFN",
                                        "order_shipment_service_level_category": "shipment_servicelevel_is_expedited",
                                        "order_order_total_currency_code": "order_currency_is_EUR",
                                        "order_shipping_address_country_code": "shipping_address_is_DE"}, 
                                        inplace=True)


        # one hot encoding
        # amazon_sales_clean = one_hot_encode(amazon_sales_clean, one_hot_encoding) -> Currently not used, because the number of payment methods is not fixed, so data structure could vary
        # only keep the not cancelled orders
        amazon_sales_clean = amazon_sales_clean[amazon_sales_clean["order_order_status"] != "Canceled"] # Pending is still inlcuded, because it is the responsibility of the seller to integrate this information in the inventory data
        amazon_sales_clean.drop(columns=["order_order_status"], inplace=True)

        # final fillna
        amazon_sales_clean.fillna(0, inplace=True)
        # groupby / aggregate sales by the date order_purchase_date?
        aggregation_dict = {
            'lineitem_quantity_ordered': 'sum',
            'lineitem_shippingprice_amount': 'mean',
            'lineitem_shippingtax_amount': 'mean',
            'lineitem_shippingdiscount_amount': 'mean',
            'lineitem_shippingdiscounttax_amount': 'mean',
            'lineitem_item_price_amount': 'mean',
            'lineitem_item_tax_amount': 'mean',
            'lineitem_promotion_discount_amount': 'mean',
            'lineitem_promotion_discount_tax_amount': 'mean',
            #'lineitem_buyer_info_giftwrap_price_amount': 'mean',
            #'lineitem_buyer_info_giftwrap_tax_amount': 'mean',
            'order_is_prime': 'mean',
            'order_is_premium_order': 'mean',
            'order_is_global_express_enabled': 'mean',
            'lineitem_is_gift': 'mean',
            'lineitem_has_promotion_ids': 'mean',
            'order_automatedshipsettings_hasautomatedshipsettings': 'mean',
            'lineitem_condition_is_new': 'mean',
            'order_is_business_order': 'mean',
            'fullfillment_channel_is_AFN': 'mean',
            'shipment_servicelevel_is_expedited': 'mean',
            'order_currency_is_EUR': 'mean',
            'shipping_address_is_DE': 'mean',
            'delivery_time_to_consumer': 'mean'
        }

        # adjust dtype for accurate aggregation
        for k in aggregation_dict.keys():
            amazon_sales_clean[k].astype("float64")

        # clean concatenated values
        for k in aggregation_dict.keys():
            amazon_sales_clean[k] = amazon_sales_clean[k].apply(clean_concatenated_values)

        # group all "to group" columns with respective aggregation functions
        amazon_sales_aggregated = amazon_sales_clean.groupby(["order_marketplace_id",
                                                                    "lineitem_asin",
                                                                    "lineitem_seller_sku",
                                                                    "order_purchase_date"],
                                                                    as_index=False).agg(aggregation_dict)
        # dtype transformation and reduction

        # Creating dtype mapping dictionary
        dtypes_dict = {col: 'str' if col in ['order_marketplace_id', 'lineitem_asin', 'lineitem_seller_sku'] else 
                    'datetime' if col in ['order_purchase_date'] else 
                    'int64' if col in ['lineitem_quantity_ordered_sum'] else
                    'float64' for col in amazon_sales_aggregated.columns}

        # Transforming dtypes
        amazon_sales_aggregated = transform_dtypes(amazon_sales_aggregated, dtypes_dict, datacollection_startdate=np.nan)

        # # Reducing dtypes to save memory
        # #amazon_sales_aggregated = reduce_mem_usage(amazon_sales_aggregated, verbose=True)
        # # rename all columns to match data structure
        # rename_dict = {'order_marketplace_id': 'marketplace_id',
        #                 'lineitem_asin': 'variant_asin',
        #                 'lineitem_seller_sku': 'variant_sku',
        #                 'order_purchase_date': 'daydate',
        #                 'lineitem_quantity_ordered': 'lineitems_quantity',
        #                 'lineitem_item_price_amount': 'lineitems_price',
        #                 'lineitem_item_tax_amount': 'lineitems_tax',
        #                 'lineitem_promotion_discount_amount': 'lineitems_discountallocations_amount',
        #                 'lineitem_promotion_discount_tax_amount': 'lineitems_discountallocations_amount_tax',
        #                 'lineitem_shippingprice_amount': 'lineitems_shippingprice',
        #                 'lineitem_shippingtax_amount': 'lineitems_shippingtax',
        #                 'lineitem_shippingdiscount_amount': 'lineitems_shippingdiscount',
        #                 'lineitem_shippingdiscounttax_amount': 'lineitems_shippingdiscount_tax',
        #                 # 'lineitem_buyer_info_giftwrap_price_amount': 'lineitems_giftwrap_price',
        #                 # 'lineitem_buyer_info_giftwrap_tax_amount': 'lineitems_giftwrap_tax',
        #                 'order_is_prim': 'order_is_prime',
        #                 'order_is_premium_order': 'order_is_premium_order',
        #                 'order_is_global_express_enabled': 'order_is_global_express_enabled',
        #                 'lineitem_is_gift': 'lineitem_is_gift',
        #                 'lineitem_has_promotion_ids': 'lineitem_has_promotion_ids',
        #                 'order_automatedshipsettings_hasautomatedshipsettings': 'order_automatedshipsettings_hasautomatedshipsettings',
        #                 'lineitem_condition_is_new': 'lineitem_condition_is_new',
        #                 'order_is_business_order': 'order_is_business_order',
        #                 'fullfillment_channel_is_AFN': 'fullfillment_channel_is_AFN',
        #                 'shipment_servicelevel_is_expedited': 'shipment_servicelevel_is_expedited',
        #                 'order_currency_is_EUR': 'order_currency_is_EUR',
        #                 'shipping_address_is_DE': 'shipping_address_is_DE',
        #                 'delivery_time_to_consumer': 'delivery_time_to_consumer'}

        # amazon_sales_aggregated = amazon_sales_aggregated.rename(columns=rename_dict)
        # # change dtype of shopify_productxdays_sceleton to merge
        # amazon_sceleton["daydate"] = pd.to_datetime(amazon_sceleton["daydate"], utc=True).dt.date
        # amazon_sales_aggregated["daydate"] = pd.to_datetime(amazon_sales_aggregated["daydate"], utc=True).dt.date        
        
        return amazon_sales_aggregated

    else:
        raise ValueError("Transformation not supported")
    

# get list of all unique product categories from sceleton. Consider that there are more than a column that indicates product categories and seperated via channel prefix
def get_unique_products(df):
    # params : df :  sceleton dataframe
    # return : list of unique products
    unique_products = []
    for col in df.columns:
        if "product_category" in col:
            unique_products.extend(df[col].unique())
    unique_products = list(set(unique_products))
    unique_products.append("Rest")
    return unique_products



def check_channel_importance(source_df, product_info, mergable_id, source_data_id_column, sales_channel):
    # params : source_df : dataframe that contains all products e.g. df_pp
    # params : id_df : dataframe that contains only products that have ids
    # return : categorize products based on sales channel to avoid double counting and merging with sceleton based on the nogl_id

    # check concatenation case :
    unique_products = set(source_df[source_data_id_column].str.lower().unique())
    # id_df contains same id across different sales channels
    sales_channel_importance = {}
    # add keys to sales_channels dict and set value to empty set
    # get columns from product_info that contains mergable_id
    mergable_columns = [col for col in product_info.columns if mergable_id in col]
    
    all_products_sets = []
    for col in mergable_columns:
        channel = col.split("_")[0]
        if channel in sales_channel:
            channel_products = set(product_info[col].str.lower().unique())
            # find intersection and store products for each channel
            sales_channel_importance[channel] = unique_products.intersection(channel_products)
            # if "nan" in sales_channel_importance[channel]:
            # remove "nan" from set
            sales_channel_importance[channel] = {product for product in sales_channel_importance[channel] if product != "nan"}
            all_products_sets.append(channel_products)

    # TODO:
    # create a new key value pair in sales_channel_importance called "common"
    # find common products across all sales channels
    common_products = set.intersection(*all_products_sets)
    sales_channel_importance['common'] = common_products
    # if common contains only "nan" then remove it and make it empty set
    sales_channel_importance['common'] = {product for product in common_products if product != "nan"}

    # remove common products from individual channels to avoid double counting
    for col in mergable_columns:
        channel = col.split("_")[0]
        if channel in sales_channel:
            sales_channel_importance[channel] = sales_channel_importance[channel].difference(common_products)

    return sales_channel_importance



def return_best_mergable_id(source_df, product_info, source_data_id_column:str=None, sales_channel:list=None):
    # source_df : df_pp # for google_ads
    # product_info : linkage_df

    # define a list of product_ids df to check
    # variant_id = shopify_variant_id
    ids = ["variant_sku","variant_barcode","variant_asin", "variant_id"]
    
    # get unique ids from corss sales channel (e.g. amazon_variant_id, shopify_variant_id)
    mergable_ids = {
        "variant_sku" : [col for col in product_info.columns if "variant_sku" in col],
        "variant_barcode" : [col for col in product_info.columns if "variant_barcode" in col],
        "variant_asin" : [col for col in product_info.columns if "variant_asin" in col],
        "variant_id" : [col for col in product_info.columns if "variant_id" in col]
    }

    IoU_scores = {
        "variant_sku" : 0,
        "variant_barcode" : 0,
        "variant_asin" : 0,
        "variant_id" : 0
    }

    mergable_products = {
        "variant_sku" : None,
        "variant_barcode" : None,
        "variant_asin" : None,
        "variant_id" : None
    }


    unique_source_products = set(source_df[source_data_id_column].unique())
    
    for id in mergable_ids:
        mergable_id = id
        sales_channel_importance = check_channel_importance(source_df=source_df, 
                                             product_info=product_info,
                                             mergable_id=mergable_id,
                                             source_data_id_column=source_data_id_column,
                                             sales_channel=sales_channel
                                             )


        mergable_products[id]  = sales_channel_importance
        # calculate IoU score
        # calculate IoU score
        for channel in sales_channel_importance:
            intersection = len(sales_channel_importance[channel])
            union = len(unique_source_products.union(sales_channel_importance[channel]))
            IoU_scores[id] += (intersection / union) * 100

    # change key name variant_id in IoU scores
    # variant_id -> shopify_variant_id
    # variant_sku -> shopify_variant_sku
    # variant_barcode -> shopify_variant_barcode
    # variant_asin -> amazon_variant_asin

    IoU_scores = {key.replace("variant_id", "shopify_variant_id").replace("variant_sku", "shopify_variant_sku").replace("variant_barcode", "shopify_variant_barcode").replace("variant_asin", "amazon_variant_asin"):value for key, value in IoU_scores.items()}


    return mergable_products, IoU_scores



def merge_with_product_info(source_df, product_info, mergable_id_products, mergable_id, source_data_id_column):
    # source_df : df_pp # for google_ads
    # product_info : linkage_df
    # mergable_id_products : mergable_products["variant_id"]
    # return : merged_df

    # get columns from product_info that contains mergable_id
    # first try to merge source_df with product_info based on mergable_id
    merged_df = source_df.merge(product_info[[mergable_id, "nogl_id"]],
        left_on=source_data_id_column,
        right_on=mergable_id,
        how="left")

    # drop mergable_id column
    merged_df.drop(columns=[mergable_id], inplace=True)

    # now try to fill up nan rows for nogl. 
    # TODO : first check nan rows for nogl and get source_data_id_column and try to find that product in mergable_id_products
    # if found then fill up nogl_id column by looking at the nogl_id column of product_info dataframe
    # e.g. if id found in amazon key set of mergable_id_products then fill up nogl_id column by looking at the amazon_"mergable_id" column of product_info dataframe

    # Identify NaN rows for 'nogl_id'
    nan_rows = merged_df[merged_df['nogl_id'].isna()]

    # For each NaN row, check if the product exists in 'mergable_id_products'
    for idx, row in nan_rows.iterrows():
        product_id = row[source_data_id_column]
        #print(product_id)
        
        for channel, ids in mergable_id_products.items():
             if product_id in ids:
                # Find the matching row in 'product_info' and get the 'nogl_id' value
                if channel == "common":
                    matched_row = product_info[product_info[f"{mergable_id}"] == product_id]
                else:
                    matched_row = product_info[product_info[f"{channel}_{mergable_id}"] == product_id]

                if not matched_row.empty:
                    matched_nogl_id = matched_row['nogl_id'].iloc[0]
                    #Update the 'nogl_id' in 'merged_df'
                    merged_df.at[idx, 'nogl_id'] = matched_nogl_id
                    break  # No need to check other channels once a match is found



    return merged_df


def product_status_mapping(df,channel=None):
    """
    Create mapping for product_status column
    "Active" -> "active"
    "Incomplete" -> "draft"
    "Inactive" -> "archived"
    """
    if channel != None:
        column_name = channel + "_product_status"
    else:
        column_name = "product_status"
    df[column_name] = df[column_name].map({"Active": "active", "Incomplete": "draft", "Inactive": "archived"})
    print(df[column_name].unique())
    return df


def filter_DF(df, filter):
    for i in filter:
        if i[1] == "==":
            df = df[df[i[0]]==i[2]]
        if i[1] == "!=":
            df = df[df[i[0]]!=i[2]]
        if i[1] == "<=":
            df = df[df[i[0]]<=i[2]]
        if i[1] == ">=":
            df = df[df[i[0]]>=i[2]]
        if i[1] == "<":
            df = df[df[i[0]]<i[2]]
        if i[1] == ">":
            df = df[df[i[0]]>i[2]]
    return df



def download_query_from_bigquery(sql):
    '''
    Accesses Google BigQuery to download data depending on the passed query
    '''

    # Set Credential location
    credentials = service_account.Credentials.from_service_account_file(
        '/opt/ml/processing/input/01_Data Source Keys/Python/BigQuery Service Account Key/noglclientprojectwefriends-336852767362.json',
    )
    
    # Download Data
    df = pandas_gbq.read_gbq(sql, project_id="noglclientprojectwefriends", credentials=credentials)
    df.drop_duplicates(inplace=True)

    return df


def load_insights(section):
    
    # adjust section variable in FROM clause if specific schema should be loaded, now its loading dynamically based on client name
    df = download_query_from_bigquery("SELECT DISTINCT ad_id,clicks, date_start, frequency, impressions, inline_post_engagements , link_clicks, reach, unique_clicks, spend FROM `noglclientprojectwefriends." + section.replace("-","_") + "_facebookads.insights`")
    print("Downloaded BQ data with query: " + "SELECT DISTINCT ad_id,clicks, date_start, frequency, impressions, inline_post_engagements , link_clicks, reach, unique_clicks, spend FROM `noglclientprojectwefriends." + section.replace("-","_") + "_facebookads.insights`")
    
    # drop duplicates with same ad id on date -> Unique ad id and date are necessary
    # decision based on highest impressions (thats the value that is first in the funnel, so easiest to be triggered)
    df.sort_values(by=["date_start","ad_id","impressions"], ascending=False, inplace=True)
    df = df.groupby(["date_start","ad_id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("insights_df shape:", df.shape)

    return df


def load_ads():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.ads`")
    
    # drop duplicates with same ad id on date -> Unique ad id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["campaign_id","adset_id","id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["campaign_id","adset_id","id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("ads_df shape:", df.shape)
    return df


def load_adsets():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.ad_sets`")
    
    # drop duplicates with same ad id on date -> Unique adset id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["account_id", "campaign_id", "id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["account_id", "campaign_id", "id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("adsets_df shape:", df.shape)
    return df


def load_campaigns():
    
    df = download_query_from_bigquery("SELECT DISTINCT * FROM `noglclientprojectwefriends.wefriends_facebookads.campaigns`")
    
    # drop duplicates with same ad id on date -> Unique campaign id and date are necessary
    # decision based on latest load time
    df.sort_values(by=["account_id", "id", "loaded_at"], ascending=False, inplace=True)
    df = df.groupby(["account_id", "id"], as_index=False).first()
    df.reset_index(inplace=True, drop=True)
    print("campaigns_df shape:", df.shape)
    return df



def preprocess_facebook_data(df):
    '''
    Takes a dataframe containing the insights datatable from the facebook ad data and returns a dataframe containingdaily data for:
    Totals and averages for: clicks, impressions, inline_post_engagement, link_clicks, reach, unique_clicks and spend; 
    The average frequency; and the number of active ads.
    '''
    
    '''
    Assumption: Every value missing in the data after the merge with the sceleton (NA values) are filled with 0. This is because a NA value will indicate
    that there was no ad active this day. This also applies to all mean/avg values.
    '''
    
    # Drop hours and minutes from date
    df['date_start'] = pd.to_datetime(df['date_start']).dt.date
    
    # Sum up data for each day
    sum_columns = ["clicks", "date_start", "impressions", "inline_post_engagements", "link_clicks", "reach", "spend", "unique_clicks"]  # Select columns to include in dataset
    fb_data_per_day = df[sum_columns].groupby("date_start").sum()  # Sum up results for each day
    fb_data_per_day.reset_index(inplace = True)  # Restore date column
    
    # Count number of Ads per day
    fb_ad_num = df[['date_start', 'ad_id']].groupby("date_start").count()  # Count ads each day
    fb_ad_num.reset_index(inplace=True)  # Restore date column
    fb_ad_num = fb_ad_num.rename(columns = {"ad_id" : "ads_per_day"})  # Set correct feature name
    
    # Add number of ads to dataframe
    fb_data_per_day = fb_data_per_day.merge(fb_ad_num, on = "date_start", how = "outer")    
    
    # Loop over columns in sum_columns ad create a new column with the average per ad 
    sum_columns.remove("date_start")  # remove non-numerical columns from list of features to average
    for column in sum_columns:  
        fb_data_per_day[column + "_avgPerAd"] = fb_data_per_day[column] / fb_data_per_day["ads_per_day"] # Divide Total per day by Number of Ads per day
    
    # Calculate average frequency
    fb_frequency_avg = df[["date_start", "frequency"]].groupby("date_start").mean()  # Average frequency on each day
    fb_frequency_avg.reset_index(inplace=True)  # Restore date column
    fb_frequency_avg = fb_frequency_avg.rename(columns = {"frequency" : "frequency_avg"})  # Rename feature column
    fb_data_per_day = fb_data_per_day.merge(fb_frequency_avg, on = "date_start", how = "outer")  # Add new data to final dataset
    
    return fb_data_per_day


def facebook_merge_into_product_x_day_skeleton(fb_data_df, skeleton_df):
    '''
    Merges the output of preprocess_facebook_data into the product x day skeleton
    '''
    
    fb_data_df = fb_data_df.rename(columns = {"date_start" : "daydate"})  # Rename date columns 
    skeleton_df = skeleton_df[['daydate', 
                               #"variant_sku", 
                               "nogl_id"]]  # Drop superfluous columns
    skeleton_df['daydate'] = pd.to_datetime(skeleton_df['daydate']).dt.date
    skeleton_df = skeleton_df.merge(fb_data_df, how = "left", on = "daydate")   # Merge both datasets
    skeleton_df = skeleton_df.fillna(0)  # Fill all na's with zero
    
    return skeleton_df


def pick_extract(extract_campaign, extract_adset, extract_ad):
    if extract_campaign != "":
        return extract_campaign
    elif extract_adset != "":
        return extract_adset
    elif extract_ad != "":
        return extract_ad
    else:
        return "Rest"