#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename : _result.py
# Author : Tuhin Mallick

# Import necessary libraries and modules
import os, sys, pathlib
import multiprocessing
import argparse
import re
import numpy as np
import pandas as pd
from typing import List, Optional
from sqlalchemy import create_engine

# Add the current path to sys.path to import the _metrics module
src_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from _metrics import METRICS  # Import the METRICS dictionary from the _metrics module

import yaml  # Import the YAML library for reading configuration files


# Define the config function to read configurations from a YAML file
def config(section, filename='/opt/ml/processing/input/Evaluation/configAWSRDS.yaml'):
    """
    Read configuration from a YAML file.

    :param section: The section in the config file to read.
    :param filename: The name of the YAML config file.
    :return: Dictionary containing the configuration data.
    """
    with open(filename, 'r') as file:
        config_data = yaml.safe_load(file)

    if section in config_data:
        return config_data[section]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))


# Define the model_evaluation function to evaluate the model using various metrics
def model_evaluation(
    data: pd.DataFrame,
    y_true_col, y_pred_col,
    metrics_suffixes: List[str] = ["MAE", "MAPE"],
    per_row: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a model using specified metrics.

    :param data: The input DataFrame containing true and predicted values.
    :param y_true_col: The name of the column containing the true values.
    :param y_pred_col: The name of the column containing the predicted values.
    :param metrics_suffixes: A list of metric suffixes to use for evaluation.
    :param per_row: If True, compute the metric for each row in the DataFrame. If False, compute the metric for the entire DataFrame.
    :return: A DataFrame containing the evaluation results.
    """
    # Initialize the list of metrics
    metrics = []
    for name in metrics_suffixes:
        if name not in METRICS:
            raise ValueError(f"No metric of name: {name}")
        metrics.append(METRICS[name]())

    if per_row:
        # Make a copy of the input DataFrame
        result_df = data.copy()

        # Calculate the metric for each row in the DataFrame
        for metric in metrics:
            print("Calculating Metric: ", metric.name)
            for index, row in data.iterrows():
                y_true = np.array(row[y_true_col])
                y_pred = np.array(row[y_pred_col])

                # Calculate the metric result for the current row
                eval_result = metric(y_pred, y_true)

                # Add the metric result as a new column to the result DataFrame
                result_df.at[index, metric.name] = eval_result

        return result_df
    else:
        # Calculate the metric for the entire DataFrame
        evaluation_metrics = {key: 0 for key in metrics_suffixes}
        for metric in metrics:
            eval_result = metric(
                np.array(data[y_pred_col]).flatten(),
                np.array(data[y_true_col]).flatten()
            )
            evaluation_metrics[metric.name] = eval_result

        return evaluation_metrics

# Define the fetch_data function to fetch data from the database
def fetch_data(
    engine,
    y_true_col,
    y_pred_col,
    schema_name="forecasts",
    table_name_pattern="forecasts_28days_incl_history",
    use_filtered_tables=False,
):
    """
    Fetch data from the database based on the provided schema and table name pattern.

    :param engine: SQLAlchemy engine object for connecting to the database.
    :param y_true_col: The name of the column containing the true values.
    :param y_pred_col: The name of the column containing the predicted values.
    :param variant_id: Variant ID to filter the data.
    :param schema_name: The name of the schema in the database.
    :param table_name_pattern: The pattern to match table names in the schema.
    :return: A dictionary containing the fetched data, keyed by table name.
    """
    # Get table names in the schema
    with engine.connect() as connection:
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema_name}';
        """
        tables = pd.read_sql_query(query, connection)["table_name"].tolist()

    # Filter table names based on the desired pattern
    if use_filtered_tables: 
        filtered_tables = [table for table in tables if re.match(f"{table_name_pattern}_model_\d+", table) or table == table_name_pattern]
    else:
        filtered_tables = tables

    # Fetch data from each table
    datasets = {}
    for table in filtered_tables:
        with engine.connect() as connection:
            query = f"""
            SELECT f."SC_product_category_number", f."SC_product_category", f."SC_product_id", f."SC_variant_sku", f."SC_variant_id", f."TVKC_daydate", f."TVUR_shopify_lineitems_quantity", f."NOGL_forecast_q0", f."NOGL_forecast_q1", f."NOGL_forecast_q2", f."NOGL_forecast_q3", f."NOGL_forecast_q4", f."NOGL_forecast_q5", f."NOGL_forecast_q6"
            FROM {schema_name}.{table} f
            WHERE f."{y_pred_col}" > 0; 
            
            """

            data = pd.read_sql_query(query, connection)
            datasets[table] = data

    return datasets

# Define the process_dataset function to transform and evaluate a dataset
def process_dataset(item, y_true_col, metric_keys):
    """
    Transform and evaluate a dataset.

    :param item: A tuple containing the table name and the dataset.
    :param y_true_col: The name of the column containing the true values.
    :param metric_keys: A list of metric keys to use for evaluation.
    :return: A DataFrame containing the transformed and evaluated data.
    """
    table, dataset = item

    # Transform the dataset
    dataset = dataset.applymap(lambda x: x.flatten()[0] if isinstance(x, np.ndarray) and x.ndim > 1 else x)
    dataset['SC_model_type'] = 'Top Seller'
    dataset_melted = pd.melt(
        dataset,
        id_vars=['SC_product_category_number', 'SC_product_category', 'SC_product_id', 'SC_variant_sku', 'SC_variant_id', 'TVKC_daydate', 'TVUR_shopify_lineitems_quantity', 'SC_model_type'],
        var_name='quantile',
        value_name='forecast_value'
    )
    dataset_melted['quantile'] = dataset_melted['quantile'].str.extract('(\d+)').astype(int)
    dataset_melted = dataset_melted.sort_values(['TVKC_daydate', 'SC_product_id', 'quantile']).reset_index(drop=True)

    return model_evaluation(dataset_melted, y_true_col, y_pred_col= "forecast_value", metrics_suffixes = metric_keys, per_row = True)

def main(
    client_name: str,
    schema: str,
    y_true_col: str,
    y_pred_col: str,
    metric_keys: List[str] = None,
    save_csv: bool = False,
    use_old: bool = True,
    use_filtered_tables: bool = False,
):
    """
    The main function that fetches data, processes it, and saves the results in a database.

    :param client_name: The name of the client (section in the config file).
    :param schema: The name of the schema in the database.
    :param y_true_col: The name of the column containing the true values.
    :param y_pred_col: The name of the column containing the predicted values.
    :param metric_keys: A list of metric keys to use for evaluation (optional).
    """
    if metric_keys is None:
        metric_keys = ["MAPE", "MAE", "SMAPE", "MSE", "P50", "P90", "ND", "Accuracy"]
    params = config(section=client_name)
    engine = create_engine(
        f"postgresql://{params.get('user')}:{params.get('password')}@{params.get('host')}:5432/{params.get('database')}",
        echo=False
    )

    if use_old:
        datasets = fetch_data(engine, y_true_col, y_pred_col, schema_name=schema)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_dataset, [((table, dataset), y_true_col, metric_keys) for table, dataset in datasets.items()])
        
        # Combine the results
        combined_results = pd.concat(results, ignore_index=True)
        # Sort the combined DataFrame
        combined_results = combined_results.sort_values(['TVKC_daydate', 'SC_variant_id']).reset_index(drop=True)
    else:
        pass


    if save_csv:
        # Save the updated DataFrame as a CSV file if the table flag is set
        combined_results.to_csv('output.csv', index=False)
    else:
        # Upload the DataFrame
        combined_results.to_sql(
            "model_evaluation",
            con=engine,
            schema='forecasts_evaluation',
            if_exists='replace',
            index=False
        )
    
    # Display the transformed DataFrame
    print(combined_results)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model evaluation script.')
    parser.add_argument('--client_name', type=str, required=True, help='Client name (section in the config file).')
    parser.add_argument('--schema', type=str, required=True, help='Schema name in the database.')
    parser.add_argument('--y_true_col', type=str, default='lineitems_quantity', help='Column name for y_true values.')
    parser.add_argument('--y_pred_col', type=str, default='NOGL_forecast_q3', help='Column name for y_pred values.')
    parser.add_argument('--metric_keys', type=str, nargs='*', help='List of metric key suffixes to use for evaluation.')
    parser.add_argument('--save_csv', type=bool, default= False, help='Save the updated DataFrame as a CSV file if the table flag is set')
    parser.add_argument('--use_old', type=bool, default= True , help='Use old data')
    parser.add_argument('--use_filtered_tables', type=bool, default= False , help='Use use_filtered_tables')

    args = parser.parse_args()
    main(   
        client_name=args.client_name,
        schema=args.schema,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        metric_keys=args.metric_keys,
        save_csv=args.save_csv,
        use_old=args.use_old,
        use_filtered_tables=args.use_filtered_tables
        )
