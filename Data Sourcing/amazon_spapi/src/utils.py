# external imports
import pandas as pd
import time
import os, sys, pathlib
import time
import chardet
import requests
import io
from sp_api.api import Reports
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType

# internal imports
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def map_item_condition(condition_number: int, case: str = "upper") -> str:
    """
    Maps the item condition number to the corresponding string representation.

    Parameters:
    condition_number (int): The item condition number.
    to_lower (bool): Flag to indicate whether to convert the string to lowercase. Defaults to False.

    Returns:
    condition (str): The mapped item condition string.
    """
    condition_map = {
        1: "USED_LIKE_NEW",
        2: "USED_VERY_GOOD",
        3: "USED_GOOD",
        4: "USED_ACCEPTABLE",
        5: "COLLECTIBLE_LIKE_NEW",
        6: "COLLECTIBLE_VERY_GOOD",
        7: "COLLECTIBLE_GOOD",
        8: "COLLECTIBLE_ACCEPTABLE",
        9: "NOT_USED",
        10: "REFURBISHED",
        11: "NEW"
    }
    # https://support.channelengine.com/hc/en-us/articles/4409485117469-Amazon-condition-mappings
    # CONDITION_TYPES = {
    #     "new_new": 11,
    #     "new_open_box": 9,
    #     "new_oem": 9,
    #     "refurbished_refurbished": 10,
    #     "used_like_new": 1,
    #     "used_very_good": 2,
    #     "used_good": 3,
    #     "used_acceptable": 4,
    #     "collectible_like_new": 5,
    #     "collectible_very_good": 6,
    #     "collectible_good": 7,
    #     "collectible_acceptable": 8,
    #     "club_club": 9
    # }

    condition = condition_map.get(condition_number, "NEW")
    if case == "lower":
        condition = condition.lower()
    elif case == "capital":
        condition = condition.capitalize()
    elif case == "upper":
        condition = condition
    return condition

def denest_response_json_to_df(json):
    """
    This function converts a nested JSON object into a flat pandas DataFrame.

    Parameters:
    json (list or dict): Input JSON object which can be a list or a dictionary.

    Returns:
    df (pd.DataFrame): Output DataFrame with flattened structure.
    """
    # Check the type of the input JSON
    if isinstance(json, list):
        # If it's a list, convert it directly into a DataFrame
        df = pd.DataFrame(json)
    elif isinstance(json, dict):
        # If it's a dictionary, convert each key-value pair into a pandas Series and then form a DataFrame
        df = pd.DataFrame({k: pd.Series(v) for k, v in json.items()})
    else:
        print("Payload is neither a list nor a dictionary.")
        return None

    # Identify the columns which have nested structure (either list or dictionary)
    nested_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (dict, list))]

    # Recursively flatten the nested columns until there are no nested columns left
    while len(nested_cols) != 0:
        df = flatten(df, nested_cols)
        nested_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (dict, list))]

    return df

def flatten(data, col_list):
    """
    This function flattens nested columns in a dataframe.

    Parameters:
    data (pd.DataFrame): Input dataframe with nested structure in columns.
    col_list (list): List of columns with nested structure.

    Returns:
    data (pd.DataFrame): Output dataframe with flattened structure in columns.
    """
    # Loop over each column in the list of nested columns
    for column in col_list:
        # Check if the nested structure is a list
        if isinstance(data[column].iloc[0], list):
            # If it's a list, explode the list into separate rows and then normalize the dictionaries into a DataFrame
            flatten_column = pd.json_normalize(data[column].explode())
            # Update the column names to reflect the nested structure
            flatten_column.columns = [f"{column}_{subcolumn}" for subcolumn in flatten_column.columns]
            # Drop the original nested column and join the flattened DataFrame with the original DataFrame
            data = data.drop(column, axis=1).join(flatten_column)
        else:
            # If it's a dictionary, normalize it into a DataFrame
            flatten_column = pd.json_normalize(data[column])
            # Update the column names to reflect the nested structure
            flatten_column.columns = [f"{column}_{subcolumn}" for subcolumn in flatten_column.columns]
            # Drop the original nested column and concatenate the flattened DataFrame with the original DataFrame
            data = pd.concat([data, flatten_column], axis=1).drop(column, axis=1)
    return data

def drop_overlapping_columns(df1, df2, columns_to_keep):
    """
    Drop the overlapping columns from the second dataframe except for those specified to be kept.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe. This dataframe is not modified.
    df2 : pandas.DataFrame
        The second dataframe. Overlapping columns with df1 will be dropped from this dataframe.
    columns_to_keep : list of str
        The list of column names to be kept in df2 even if they overlap with df1.

    Returns
    -------
    df1 : pandas.DataFrame
        The first dataframe. This dataframe is not modified.
    df2 : pandas.DataFrame
        The second dataframe with overlapping columns dropped except for those specified to be kept.
    """
    # Get a list of overlapping column names
    overlapping_columns = df1.columns.intersection(df2.columns)

    # Remove the columns to keep from the list of overlapping columns
    overlapping_columns = [col for col in overlapping_columns if col not in columns_to_keep]

    # Drop overlapping columns from the second dataframe
    df2 = df2.drop(columns=overlapping_columns)

    return df1, df2