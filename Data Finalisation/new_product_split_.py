import logging
import pandas as pd
import numpy as np
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import traceback
from _product_analysis import ProductAnalysis


def validate_data(data):
    required_columns = ["SC_variant_id",  "TVKC_daydate", "TVKR_variant_RRP"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Data is missing required column: {col}")

def zero_plateau_categorization(analysis):
    zero_plateau = analysis.compute_zero_plateau()

    complete_zero_ids = zero_plateau[zero_plateau == 100].index.tolist()
    zero_plateau = zero_plateau.drop(complete_zero_ids)

    high_plateau_ids = zero_plateau[zero_plateau > 90].index.tolist()
    mid_plateau_ids = zero_plateau[(zero_plateau <= 90) & (zero_plateau > 50)].index.tolist()
    low_plateau_ids = zero_plateau[zero_plateau <= 50].index.tolist()

    return low_plateau_ids, mid_plateau_ids, high_plateau_ids, complete_zero_ids

def cov_categorization(analysis, quantile_loss_ids, mid_range_ids, naive_forecast_ids_initial, complete_zero_ids):
    cov_values = analysis.compute_cov()
    cov_flags = analysis.flag_cov(cov_values)

    low_volatility_ids_set = set(cov_flags[cov_flags == "Low Volatility"].index.tolist()) - set(complete_zero_ids)
    naive_forecast_ids_initial_set = set(naive_forecast_ids_initial)

    # Ensure that low_volatility_ids do not contain any IDs that are already in naive_forecast_ids_initial
    low_volatility_ids_set -= naive_forecast_ids_initial_set
    
    naive_forecast_ids_from_cov = list(naive_forecast_ids_initial_set | low_volatility_ids_set)

    quantile_loss_ids = list(set(quantile_loss_ids) - low_volatility_ids_set)
    mid_range_ids = list(set(mid_range_ids) - low_volatility_ids_set)

    return quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_cov

def product_ranking_categorization(analysis, quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_cov, complete_zero_ids):
    importance_flags = analysis.flag_importance()
    
    low_importance_ids_set = set(importance_flags[importance_flags == "Low Importance"].index.tolist()) - set(complete_zero_ids)
    naive_forecast_ids_from_cov_set = set(naive_forecast_ids_from_cov)

    # Ensure that low_importance_ids do not contain any IDs that are already in naive_forecast_ids_from_cov
    low_importance_ids_set -= naive_forecast_ids_from_cov_set
    
    naive_forecast_ids_from_rank = list(naive_forecast_ids_from_cov_set | low_importance_ids_set)

    quantile_loss_ids = list(set(quantile_loss_ids) - low_importance_ids_set)
    mid_range_ids = list(set(mid_range_ids) - low_importance_ids_set)

    return quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_rank

def recent_products_categorization(analysis, days_threshold=60):
    """Categorizes products as recent based on their first sale date.
    
    Args:
    - analysis (ProductAnalysis): An instance of the ProductAnalysis class with loaded data.
    - days_threshold (int, optional): Number of days to consider a product as recent. Defaults to 90.
    
    Returns:
    - List[int]: A list of product IDs categorized as recent.
    """
#     import pdb;pdb.set_trace()
    # Extract data from the analysis object
    data = analysis.data
    
    # Convert the daydate column to datetime
    data['TVKC_daydate'] = pd.to_datetime(data['TVKC_daydate'])
    
    # Get the maximum date in the dataset (the latest date)
    max_date = data['TVKC_daydate'].max()
    
    # Calculate the first date of each product
    product_first_date = data.groupby('SC_variant_id')['TVKC_daydate'].min()
    
    # Identify products that have a difference between max_date and their first date less than days_threshold
    recent_products = product_first_date[product_first_date + pd.Timedelta(days=days_threshold) > max_date].index.tolist()
    
    return recent_products


def log_info_stage(stage, quantile_loss_ids, mid_range_ids, naive_forecast_ids, complete_zero_ids):
    print("="*30)
    print(f"Stage: {stage}")
    print("-"*30)
    
    print(f"Quantile Loss IDs Count: {len(quantile_loss_ids)}")
    if len(quantile_loss_ids) < 10:
        print(f"Quantile Loss IDs: {quantile_loss_ids}")

    print(f"Mid Range IDs Count: {len(mid_range_ids)}")
    if len(mid_range_ids) < 10:
        print(f"Mid Range IDs: {mid_range_ids}")

    print(f"Naive Forecast IDs Count: {len(naive_forecast_ids)}")
    if len(naive_forecast_ids) < 10:
        print(f"Naive Forecast IDs: {naive_forecast_ids}")

    print(f"Complete Zero IDs Count: {len(complete_zero_ids)}")
    if len(complete_zero_ids) < 10:
        print(f"Complete Zero IDs: {complete_zero_ids}")
    print("="*30)

def categorize_products(param_store_path, days, data_path=None, data=None, target = None):
    try:
        analysis = ProductAnalysis(param_store_path, days, data_path=data_path, data=data, target=target)
        analysis.load_data()
        validate_data(analysis.data)
        
        # Zero Plateau Categorization
        quantile_loss_ids, mid_range_ids, naive_forecast_ids_initial, complete_zero_ids = zero_plateau_categorization(analysis)
        log_info_stage("Zero Plateau", quantile_loss_ids, mid_range_ids, naive_forecast_ids_initial, complete_zero_ids)

        # COV Categorization
        quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_cov = cov_categorization(analysis, quantile_loss_ids, mid_range_ids, naive_forecast_ids_initial, complete_zero_ids)
        log_info_stage("COV", quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_cov, complete_zero_ids)

        # Rank Categorization
        quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_rank = product_ranking_categorization(analysis, quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_cov, complete_zero_ids)
        log_info_stage("Rank", quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_rank, complete_zero_ids)
        
        # Recent Products Categorization
        recent_naive_ids = recent_products_categorization(analysis)
        naive_forecast_ids_from_rank = list(set(naive_forecast_ids_from_rank) | set(recent_naive_ids))
        quantile_loss_ids = list(set(quantile_loss_ids) - set(recent_naive_ids))
        mid_range_ids = list(set(mid_range_ids) - set(recent_naive_ids))
        log_info_stage("Recent Products", quantile_loss_ids, mid_range_ids, naive_forecast_ids_from_rank, complete_zero_ids)

        quantile_loss_data = data[data['SC_variant_id'].isin(quantile_loss_ids)]
        tweedie_loss_data = data[data['SC_variant_id'].isin(mid_range_ids)]
        naive_forecast_data = data[data['SC_variant_id'].isin(naive_forecast_ids_from_rank)]
        
        total_unique_ids = data['SC_variant_id'].nunique()
        
        quantile_unique_ids = quantile_loss_data['SC_variant_id'].nunique()
        tweedie_unique_ids = tweedie_loss_data['SC_variant_id'].nunique()
        naive_unique_ids = naive_forecast_data['SC_variant_id'].nunique()
        
        quantile_loss_percentage = (quantile_unique_ids / total_unique_ids) * 100
        tweedie_loss_percentage = (tweedie_unique_ids / total_unique_ids) * 100
        naive_forecast_percentage = (naive_unique_ids / total_unique_ids) * 100
        removed_percentage = (len(complete_zero_ids) / total_unique_ids) * 100
        
        print(f"Unique variant IDs for Quantile Loss: {quantile_unique_ids}, {quantile_loss_percentage:.2f}% of total")
        print(f"Unique variant IDs for Tweedie Loss: {tweedie_unique_ids}, {tweedie_loss_percentage:.2f}% of total")
        print(f"Unique variant IDs for Naive Forecast: {naive_unique_ids}, {naive_forecast_percentage:.2f}% of total")
        print(f"Products with 100% Zero Plateau (Removed): {len(complete_zero_ids)}, {removed_percentage:.2f}%")

        return {
            'quantile_loss_data': quantile_loss_data,
            'tweedie_loss_data': tweedie_loss_data,
            'naive_forecast_data': naive_forecast_data
        }
        
    except Exception as e:
        error_message = f"An error occurred during categorization: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        raise

