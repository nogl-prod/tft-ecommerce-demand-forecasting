import pandas as pd
import numpy as np
# from line_profiler import LineProfiler
from typing import List
# from _database_manager import DatabaseManager
# from one_drive_client import OneDriveAPI

class ForecastDataUnbundler:
    def __init__(self, bundle_matrix: str, target_columns: List[str]):
        """
        Initialize the SalesDataUnbundler object.
        
        Parameters:
        bundle_matrix_path (str): The path to the bundle matrix Excel file.
        target_columns (list of str): The list of columns that contain the target metrics to be used in the calculation.
        """
        # Read in bundle matrix
        self.matrix = bundle_matrix
        # Since df does not have variant_id but SC_variant_id
        self.matrix = self.matrix.rename(columns={'variant_id':'SC_variant_id'})
        self.matrix = self.matrix.set_index("SC_variant_id")
        # Get names of single items
        self.items = self.matrix.columns
        # Get names of bundles
        self.bundles = self.matrix.index
        # Target columns
        self.target_columns = target_columns
        # Bundle columns
        self.bundle_columns = [f"{item}~{column}~bundles" for item in self.items for column in self.target_columns]
    
    def _get_bundle_metrics(self, df_bundles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the metrics for each item and column using the bundle matrix.
        
        Parameters:
        df_bundles (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
        
        Returns: 
        pandas.DataFrame: The dataframe with the calculated metrics for each item and column.
        """
        df_bundles_new = pd.DataFrame(columns=self.bundle_columns)
        # Calculating the metrices for each item and column using the bundle matrix
        for column in self.target_columns:
            for item in self.items:
                if column in df_bundles.columns and item in df_bundles.columns:
                    # import pdb; pdb.set_trace()
                    df_bundles_new[str(item) + "~" + str(column) + "~bundles"] = (df_bundles[column].fillna(0).astype(float)) * (df_bundles[item].fillna(0).astype(float))
        return df_bundles_new
    
    def _get_single_item_metrics(self, df_bundles: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the metrics for single items using the calculated bundle metrics.
        
        Parameters:
        df_bundles (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
        
        Returns:
        pandas.DataFrame: The dataframe with the calculated metrics for each single item.
        """
        # Creating a list of dates to loop over
        dates = df_bundles["TVKC_daydate"].unique()
        # Creating a df with the single items
        df_single = df.loc[df["isBundle"]==False].copy()
        # Get output columns
        df_bundles.set_index("SC_variant_id", inplace=True)
        output_columns = df_bundles.iloc[:,-len(self.target_columns)*len(self.items):].columns
        # Attributing the metrics on the single items
        quantiles = df_single['quantile'].unique()  # get unique quantiles
        for date in dates:
            for column in output_columns:
                item = column.split("~")[0]
                columnstr = column.split("~")[1]
                for quantile in quantiles:
                    mask = (
                        (df_single["SC_variant_id"]==item) 
                        & (df_single["TVKC_daydate"]==date) 
                        & (df_single['quantile'] == quantile)
                    )
                    # print(df_single.loc[mask, columnstr])  # This will print the subset where the mask is True
                    df_single.loc[mask, columnstr] = df_bundles[column].loc[
                        (df_bundles["TVKC_daydate"]==date) 
                        & (df_bundles['quantile'] == quantile)
                    ].sum()
        df_single = df_single.rename(columns={'forecast_values': 'forecast_values_bundle'})
        df_single = df_single.reindex(df.index, fill_value=0)
        final_df = df.copy()
        final_df['forecast_values_bundle'] = df_single['forecast_values_bundle']
        return final_df

    def unbundle_sales_data(self,  df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the unbundling of sales data by attributing the sales of bundles to the single items building the bundles.
    
        Parameters:
        clientname (str): The name of the client.
        df (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
        quantity_df (pandas.DataFrame): The inventory quantities for each product.
    
        Returns:
        pandas.DataFrame: The final dataframe with the unbundled sales data and the inventory quantities.
        """
        try:
            # import pdb; pdb.set_trace()
            df['SC_variant_id'] = df['SC_variant_id'].astype(str)
            self.matrix.index = self.matrix.index.astype(str)
            self.bundles = [str(bundle) for bundle in self.bundles]
            # Adding a boolean as flag if item is bundle or not
            df['isBundle'] = np.where(df['SC_variant_id'].isin(self.bundles), True, False)
            # Merge df with bundle matrix
            # df_bundles = df.merge(self.matrix, on=["SC_variant_id"], how='left')
            df_bundles = df.merge(self.matrix, on=["SC_variant_id"], how='left')
            
            # Fill NaN values in all columns with 0
            df_bundles.fillna(0, inplace=True)
        except KeyError as e:
            raise KeyError(f"Error merging dataframes. Ensure the specified merge columns are present in both dataframes. Details: {e}")
        # Calculate bundle metrics
        df_bundles_new = self._get_bundle_metrics(df_bundles)
        # Merge bundle metrics with original bundle df
        org_bundle_df = df_bundles[df_bundles["isBundle"]==True]
        df_bundles = org_bundle_df.merge(df_bundles_new, left_index=True, right_index=True)
        # Calculate single item metrics
        final_df = self._get_single_item_metrics(df_bundles, df)
        for column in self.target_columns:
            final_df[column+"_total"] = final_df.apply(lambda row: float(row[column] or 0.0) + float(row[column+"_bundle"] or 0.0), axis=1)
        # Sort the columns of final_df alphabetically
        sorted_cols = sorted(final_df.columns)
        final_df = final_df[sorted_cols]
       
        final_df.fillna(0, inplace=True)
        return final_df
    
    
if __name__ == "__main__":
    try:
        client_name='stoertebekker'
        results = pd.read_csv(r"C:\Users\Tuhin Mallick\Downloads\forecast.csv")

        bundle_matrix = pd.read_excel(r"C:\Users\Tuhin Mallick\Downloads\Bundel-Product-Matrix-stoertebekker_testing.xlsx")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading input files. Ensure the specified file paths are correct. Details: {e}")
    target_columns = [  'forecast_values']
    unbundler = ForecastDataUnbundler(bundle_matrix, target_columns)
    forecast_bundling = unbundler.unbundle_sales_data(df=results)
    forecast_bundling.to_csv("forecast_bundling.csv")
