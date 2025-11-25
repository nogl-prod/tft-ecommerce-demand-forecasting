# Import required modules and custom classes
import logging
import pandas as pd
import numpy as np
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from parameter_store_client import EC2ParameterStore


def fetch_parameters_from_store(path) -> dict:
    """Fetches parameters from the AWS Parameter Store using a custom EC2ParameterStore client."""
    try:
        store = EC2ParameterStore()
        parameters = store.get_parameters_by_path(path)
        return parameters
    except Exception as e:
        logging.error(f"Error occurred while fetching parameters from store: {e}")
        raise

class ProductAnalysis:
    """
    A class used to perform various analyses on a product dataset.

    ...

    Attributes
    ----------
    data_path : str
        the path to the dataset file
    data : pd.DataFrame
        the loaded dataset
    thresholds : dict
        the dictionary containing the threshold values for analysis

    Methods
    -------
    ... (rest of the class documentation)
    """

    def __init__(self, param_store_path, days=None, reduction="median", data_path=None, data=None, target = None):
        """
        Parameters
        ----------
        data_path : str
            The path to the dataset file.
        param_store_path : str
            The path in the AWS Parameter Store where the threshold values are stored.
        days : int, optional
            Number of days from the last date to include in the data. If None, all data is used.
        reduction : str
            The method to aggregate seasonal values (e.g., 'median', 'mean', 'sum', etc.)
        """
        self.data_path = data_path
        self.thresholds = fetch_parameters_from_store(param_store_path)
        self.data_path = data_path
        self.data = data
        self.days = days
        self.reduction = reduction
        self.target = target
        
    def load_data(self):
        """Loads the dataset from the provided path and aggregates sales on a daily basis."""
        try:
            if self.data_path:
                chunksize = 10 ** 6  # adjust this value depending on your available memory
                chunks = []
    
                for chunk in pd.read_csv(self.data_path, chunksize=chunksize, index_col=0):
                    chunks.append(chunk)
    
                # concatenate all chunks
                self.data = pd.concat(chunks, axis=0)

            # Further groupby in case there are any overlaps from chunk processing
#             self.data = self.data.groupby(['created_at', 'SC_variant_id']).sum().reset_index()
            #Truncate the data to the specified days after loading
            if self.days:
                self.data = self.truncate_data_to_days(self.data, days=self.days)
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            self.data = None
    
    @staticmethod
    def truncate_data_to_days(data, date_column='TVKC_daydate', days=None):
        if days is None:
            return data

        # Ensure the date_column is in datetime format
        data[date_column] = pd.to_datetime(data[date_column])

        end_date = pd.Timestamp.now().normalize()  # Current date without the time component
        print(end_date)
        start_date = end_date - pd.Timedelta(days=days)
        print(start_date)

        truncated_data = data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]

        return truncated_data

            
    @staticmethod
    def trim_series_from_first_non_zero(series):
        """Trims a time series to start from the first non-zero value."""
        first_non_zero_idx = series.ne(0).idxmax()  # Get the index of the first non-zero value
        return series.loc[first_non_zero_idx:]
    
    @staticmethod
    def remove_outliers_using_iqr(series):
        """Removes outliers from a series using the IQR method.
        
        Args:
        - series (pd.Series): The input series from which outliers need to be removed.
        
        Returns:
        - pd.Series: Series after removing outliers.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]


    
    def compute_zero_plateau(self):
        """Computes the percentage of zero sales values for each product, starting from the first non-zero value."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        def calculate_zero_pct(series):

            trimmed_series = ProductAnalysis.trim_series_from_first_non_zero(series)
            return (trimmed_series == 0).mean() * 100

        zero_plateau = self.data.groupby('SC_variant_id')[self.target].apply(calculate_zero_pct)
        return zero_plateau
    
    def flag_zero_plateau(self, zero_plateau):
        """Flags products based on zero plateau values and the threshold."""
        zero_plateau_threshold = float(self.thresholds.get("zero_plateau_threshold", 50.00))
        
        flags = zero_plateau.apply(lambda x: 'High' if x > zero_plateau_threshold else 'Low')
        return flags
    
    def compute_cov(self):
        """Computes the Coefficient of Variance for each product."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        def calculate_cov(series):
            trimmed_series = self.trim_series_from_first_non_zero(series)
            filtered_series = ProductAnalysis.remove_outliers_using_iqr(trimmed_series)
            return (filtered_series.std() / filtered_series.mean()) * 100 if filtered_series.mean() != 0 else 0

        cov_values = self.data.groupby('SC_variant_id')[self.target].apply(calculate_cov)
        return cov_values


    def flag_cov(self, cov_values):
        """Flags products based on CoV values and the threshold."""
        cov_threshold = float(self.thresholds.get("cov_threshold", 30.00))
        
        flags = cov_values.apply(lambda x: 'High Volatility' if x > cov_threshold else 'Low Volatility')
        return flags
    
    def compute_cot(self):
        """Computes the Coefficient of Trend Variance for each product."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        def calculate_cot(series):
            trimmed_series = ProductAnalysis.trim_series_from_first_non_zero(series)
            filtered_series = ProductAnalysis.remove_outliers_using_iqr(trimmed_series)
            return (detrend(filtered_series).std() / filtered_series.mean()) * 100 if filtered_series.mean() != 0 else 0

        cot = self.data.groupby('SC_variant_id')[self.target].apply(calculate_cot)
        return cot

    
    def flag_cot(self):
        """Flags products with Coefficient of Trend Variance based on the threshold from the Parameter Store."""
        cot = self.compute_cot()
        cot_threshold = float(self.thresholds.get("cot_threshold", 30.00))  # Default to 30 if not provided
        flags = cot.apply(lambda x: 'High Trend Variance' if x > cot_threshold else 'Low Trend Variance')
        return flags
    
    def compute_seasonality(self, period):
        """Computes the seasonality value for the given period (weekly, monthly, yearly) using the specified reduction."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        def extract_seasonality(series):
            trimmed_series = ProductAnalysis.trim_series_from_first_non_zero(series)
            filtered_series = ProductAnalysis.remove_outliers_using_iqr(trimmed_series)
            if len(filtered_series) >= 2 * period:  # Ensure there are enough data points
                seasonal_values = seasonal_decompose(filtered_series, model='additive', period=period).seasonal
                if self.reduction == "median":
                    return seasonal_values.median()
                elif self.reduction == "mean":
                    return seasonal_values.mean()
                elif self.reduction == "sum":
                    return seasonal_values.sum()
                else:
                    raise ValueError(f"Invalid reduction method: {self.reduction}")
            else:
                logging.warning(f"Insufficient data for product with SC_variant_id {series.name}. Required: {2*period}, available: {len(filtered_series)}")
                return -1  # or some other sentinel value

        seasonality = self.data.groupby('SC_variant_id')[self.target].apply(extract_seasonality)
        return seasonality

    
    def median_mean_seasonality(self, period):
        """Determines the median and mean seasonality value for the given period."""
        seasonality = self.compute_seasonality(period)
        median_values = seasonality.apply(np.median)
        mean_values = seasonality.apply(np.mean)
        return median_values, mean_values
    
    def flag_high_seasonality(self, period):
        """Flags products with a seasonality value based on the thresholds from the Parameter Store."""
        seasonality = self.compute_seasonality(period)
        seasonality_low = float(self.thresholds.get("seasonality_low", 0.5))  # Default to 0.5 if not provided
        seasonality_high = float(self.thresholds.get("seasonality_high", 1))  # Default to 1 if not provided

        # Instead of using both median and mean, use the reduced seasonality value for flagging
        flags = (seasonality > seasonality_low) & (seasonality <= seasonality_high)

        return flags.apply(lambda x: 'High Seasonality' if x else 'Low Seasonality')

    
    def rank_products(self):
        """Ranks products by number of sales and revenue volume."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        def calculate_revenue(series):
            trimmed_series = ProductAnalysis.trim_series_from_first_non_zero(series)
            return trimmed_series['TVKR_variant_RRP'] * trimmed_series[self.target]

        self.data['revenue'] = self.data.groupby('SC_variant_id').apply(calculate_revenue)

        sales_rank = self.data.groupby('SC_variant_id')[self.target].sum().rank(ascending=False)
        revenue_rank = self.data.groupby('SC_variant_id')['revenue'].sum().rank(ascending=False)

        return sales_rank, revenue_rank

    
    def flag_importance(self):
        """Flags products based on their sales rank and revenue volume."""
        assert self.data is not None, "Data has not been loaded. Please load the data first."

        # Calculate revenue for each product
        self.data['revenue'] = self.data['TVKR_variant_RRP'] * self.data[self.target]

        # Rank products by number of sales and revenue volume
        sales_rank = self.data.groupby('SC_variant_id')[self.target].sum().rank(pct=True)
        revenue_rank = self.data.groupby('SC_variant_id')['revenue'].sum().rank(pct=True)

        # Average rank
        avg_rank = (sales_rank + revenue_rank) / 2

        # Fetch the rank threshold from the parameter store
        rank_threshold = float(self.thresholds.get("rank_threshold", 0.80))

        # Flag products based on the threshold
        flags = avg_rank.apply(lambda x: 'High Importance' if x >= rank_threshold else 'Low Importance')

        return flags

