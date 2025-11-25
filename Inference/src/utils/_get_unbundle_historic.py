import pandas as pd
import numpy as np
# from line_profiler import LineProfiler
from typing import List
# from _database_manager import DatabaseManager
# from one_drive_client import OneDriveAPI

# class SalesDataUnbundler:
#     def __init__(self, bundle_matrix: str, target_columns: List[str]):
#         """
#         Initialize the SalesDataUnbundler object.
        
#         Parameters:
#         bundle_matrix_path (str): The path to the bundle matrix Excel file.
#         target_columns (list of str): The list of columns that contain the target metrics to be used in the calculation.
#         """
#         # Read in bundle matrix
#         self.matrix = bundle_matrix
#         # Since df does not have variant_sku but SC_variant_sku
#         self.matrix = self.matrix.rename(columns={'variant_sku':'SC_variant_sku'})
#         self.matrix = self.matrix.set_index("SC_variant_sku")
#         # Get names of single items
#         self.items = self.matrix.columns
#         # Get names of bundles
#         self.bundles = self.matrix.index
#         # Target columns
#         self.target_columns = target_columns
#         # Bundle columns
#         self.bundle_columns = [f"{item}~{column}~bundles" for item in self.items for column in self.target_columns]
    
#     def _get_bundle_metrics(self, df_bundles: pd.DataFrame) -> pd.DataFrame:
#         """
#         Calculate the metrics for each item and column using the bundle matrix.
        
#         Parameters:
#         df_bundles (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
        
#         Returns: 
#         pandas.DataFrame: The dataframe with the calculated metrics for each item and column.
#         """
#         df_bundles_new = pd.DataFrame(columns=self.bundle_columns)
#         # Calculating the metrices for each item and column using the bundle matrix
#         for column in self.target_columns:
#             for item in self.items:
#                 df_bundles_new[item + "~" + column + "~bundles"] = df_bundles[column] * df_bundles[item]
#         return df_bundles_new
    
#     def _get_single_item_metrics(self, df_bundles: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Calculate the metrics for single items using the calculated bundle metrics.
        
#         Parameters:
#         df_bundles (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
        
#         Returns:
#         pandas.DataFrame: The dataframe with the calculated metrics for each single item.
#         """
#         # Creating a list of dates to loop over
#         dates = df_bundles["TVKC_daydate"].unique()
#         # Creating a df with the single items
#         df_single = df.loc[df["isBundle"]==False].copy()
#         # Get output columns
#         df_bundles.set_index("SC_variant_sku", inplace=True)
#         output_columns = df_bundles.iloc[:,-len(self.target_columns)*len(self.items):].columns
#         # Attributing the metrics on the single items
#         for date in dates:
#             for column in output_columns:
#                 item = column.split("~")[0]
#                 columnstr = column.split("~")[1]
#                 mask = (df_single["SC_variant_sku"]==item) & (df_single["TVKC_daydate"]==date)
#                 df_single.loc[mask, columnstr] = df_bundles[column].loc[df_bundles["TVKC_daydate"]==date].sum()
#         df_single = df_single.rename(columns={'forecast_values': 'forecast_values_bundle'})
#         df_single = df_single.reindex(df.index, fill_value=0)
#         final_df = df.copy()
#         final_df['forecast_values_bundle'] = df_single['forecast_values_bundle']
#         return final_df

#     def unbundle_sales_data(self,  df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Perform the unbundling of sales data by attributing the sales of bundles to the single items building the bundles.
    
#         Parameters:
#         clientname (str): The name of the client.
#         df (pandas.DataFrame): The sales data, where each row represents a single sale transaction, and the columns contain information about the date, the product sold, the quantity sold, the revenue generated, and other related metrics.
#         quantity_df (pandas.DataFrame): The inventory quantities for each product.
    
#         Returns:
#         pandas.DataFrame: The final dataframe with the unbundled sales data and the inventory quantities.
#         """
#         try:
#             # Adding a boolean as flag if item is bundle or not
#             print("DF columns",df.columns)
#             df['isBundle'] = np.where(df['SC_variant_sku'].isin(self.bundles), True, False)
#             # Merge df with bundle matrix
#             # df_bundles = df.merge(self.matrix, on=["SC_variant_sku"], how='left')
#             df_bundles = df.merge(self.matrix, on=["SC_variant_sku"], how='left')
#         except KeyError as e:
#             raise KeyError(f"Error merging dataframes. Ensure the specified merge columns are present in both dataframes. Details: {e}")
#         # Calculate bundle metrics
#         df_bundles_new = self._get_bundle_metrics(df_bundles)
#         # Merge bundle metrics with original bundle df
#         org_bundle_df = df_bundles[df_bundles["isBundle"]==True]
#         df_bundles = org_bundle_df.merge(df_bundles_new, left_index=True, right_index=True)
#         # Calculate single item metrics
#         final_df = self._get_single_item_metrics(df_bundles, df)
#         for column in self.target_columns:
#             final_df[column+"_total"] = final_df.apply(lambda row: float(row[column] or 0.0) + float(row[column+"_bundle"] or 0.0), axis=1)
#         # Sort the columns of final_df alphabetically
#         sorted_cols = sorted(final_df.columns)
#         final_df = final_df[sorted_cols]
        
#         final_df.fillna(0, inplace=True)
#         return final_df
    
# if __name__ == "__main__":
#     try:
#         client_name='stoertebekker'
#         downloader = DatabaseManager(client_name=client_name, config_filename = r'C:\Users\tuhin\Repo\Data-Processing-Deployment-Test\Inference\config\configAWSRDS.yaml')
#         results_path = "C:\\Users\\tuhin\\Downloads\\forecast.csv"
#         results = pd.read_csv(results_path)
#         onedrive_api = OneDriveAPI(client_name, config_filename = r'C:\Users\tuhin\Repo\Data-Processing-Deployment-Test\Inference\config\configOneDrive.yaml')
#         bundle_matrix = onedrive_api.download_file_by_relative_path("NOGL_shared/"+client_name+"/Bundel-Product-Matrix-"+client_name+".xlsx")
#     except FileNotFoundError as e:
#         raise FileNotFoundError(f"Error reading input files. Ensure the specified file paths are correct. Details: {e}")

#     target_columns = [  'forecast_values']
#     unbundler = SalesDataUnbundler(bundle_matrix, target_columns)
#     forecast_bundling = unbundler.unbundle_sales_data(df=results)
#     forecast_bundling.to_csv("forecast_bundling.csv")
class SalesDataUnbundler:
    def __init__(self, bundle_matrix, df_column_id, matrix_column_id, target_to_unbundle, date_column):
        self.bundle_matrix = bundle_matrix
        self.df_column_id = df_column_id
        self.matrix_column_id = matrix_column_id
        self.target_to_unbundle = target_to_unbundle
        self.date_column = date_column
        
    def map_bundle_to_product_ids(self, df):
        print("Mapping bundles to product ids...")
        dictionary = dict(zip(df[self.matrix_column_id], df[self.df_column_id]))
        self.bundle_matrix[self.matrix_column_id] = self.bundle_matrix[self.matrix_column_id].replace(dictionary)
        self.bundle_matrix.rename(columns=dictionary, inplace=True)
        self.bundle_matrix.rename(columns={self.matrix_column_id:self.df_column_id}, inplace=True)
        
    def flag_bundles(self):
        print("Flagging bundles...")
        is_bundle_df = self.bundle_matrix[self.df_column_id].reset_index()
        is_bundle_df["isBundle"] = True
        is_bundle_df.drop(columns="index", inplace=True)
        return is_bundle_df

    def merge_bundles(self, df, is_bundle_df):
        print("Merging bundles...")
        is_bundle_df[self.df_column_id] = is_bundle_df[self.df_column_id].astype(str)
        df[self.df_column_id] = df[self.df_column_id].astype(str)
        
        df_bundles = df.merge(is_bundle_df, how="left", on=self.df_column_id)
        df_bundles["isBundle"] = df_bundles['isBundle'].fillna(False)
        df_bundles = df_bundles.merge(self.bundle_matrix, how="left", on=self.df_column_id)
        df_bundles = df_bundles[df_bundles["isBundle"]]
        return df_bundles

    def calculate_bundle_sales(self, df_bundles):
        print("Calculating bundle sales...")
        bundle_columns = [col for col in df_bundles.columns if col not in [self.df_column_id, self.date_column, self.target_to_unbundle, 'isBundle']]
        for c in bundle_columns:
            df_bundles[c] = df_bundles[c] * df_bundles[self.target_to_unbundle]
        df_bundles_melted = df_bundles.drop(columns=["isBundle", self.target_to_unbundle, self.df_column_id]).melt(id_vars=[self.date_column])
        df_bundles_melted = df_bundles_melted.groupby(["variable", self.date_column], as_index=False)["value"].sum()
        df_bundles_melted.rename(columns={"variable":self.df_column_id,"value":self.target_to_unbundle+"_bundle"}, inplace=True)
        return df_bundles_melted

    def merge_and_cleanup(self, df, df_bundles_melted, is_bundle_df):
        print("Merging and cleaning up...")
        df[self.df_column_id] = df[self.df_column_id].astype(str)
        unbundled = df.merge(df_bundles_melted, how="left", on=[self.df_column_id, self.date_column])
        unbundled.fillna(0, inplace=True)
        unbundled[self.target_to_unbundle+"_total"] = unbundled[self.target_to_unbundle] + unbundled[self.target_to_unbundle+"_bundle"]
        unbundled.drop(columns=self.target_to_unbundle, inplace=True)
        is_bundle_df[self.df_column_id] = is_bundle_df[self.df_column_id].astype(str)
        unbundled = unbundled.merge(is_bundle_df, how="left", on=self.df_column_id)
        unbundled["isBundle"] = unbundled['isBundle'].fillna(False)
        print("unbundled data...", unbundled.head())
        return unbundled

    def unbundle(self, df):
        print("Unbundling...")
        self.map_bundle_to_product_ids(df)
        is_bundle_df = self.flag_bundles()
        df_sub = df[[self.date_column, self.df_column_id, self.target_to_unbundle]]
        df_bundles = self.merge_bundles(df_sub, is_bundle_df)
        df_bundles_melted = self.calculate_bundle_sales(df_bundles)
        return self.merge_and_cleanup(df_sub, df_bundles_melted, is_bundle_df)




# import pdb; pdb.set_trace()

# # Load data from local files
# results = pd.read_csv(r"C:\Users\Tuhin Mallick\Downloads\results.csv")
# # downloader = DatabaseManager(client_name = "stoertebekker", config_filename = r'C:\Users\Tuhin Mallick\OneDrive - NOGL\Documents\GitHub\Data-Processing-Deployment-Test\Inference\config\configAWSRDS.yaml')
# # shopify_sales_data = downloader.import_data_from_aws_rds(schema="transformed", table="shopify_sales")
# bundle_matrix = pd.read_csv(r"C:\Users\Tuhin Mallick\Downloads\bundle_matrix.csv")

# # Create unbundler for forecast results
# forecast_unbundler = SalesDataUnbundler(bundle_matrix=bundle_matrix.rename(columns={"variant_sku":"SC_variant_sku"}), 
#                                         df_column_id='SC_variant_id', 
#                                         matrix_column_id='SC_variant_sku', 
#                                         target_to_unbundle='forecast_values', 
#                                         date_column='TVKC_daydate')

# # Apply unbundling to forecast results
# forecast_bundle_results = forecast_unbundler.unbundle(df=results)

# # Save unbundled forecast results
# forecast_bundle_results.to_csv('unbundled_forecast_results.csv', index=False)


# # Create unbundler for historic results
# historic_unbundler = SalesDataUnbundler(bundle_matrix=bundle_matrix, 
#                                         df_column_id='variant_id', 
#                                         matrix_column_id='variant_sku', 
#                                         target_to_unbundle='lineitems_quantity', 
#                                         date_column='daydate')

# # Apply unbundling to historic results
# historic_bundle_results = historic_unbundler.unbundle(df=shopify_sales_data)
# import pdb;pdb.set_trace()
# df_concatenated = pd.concat([historic_bundle_results, shopify_sales_data], axis=0)

# # Save unbundled historic results
# historic_bundle_results.to_csv('unbundled_historic_results.csv', index=False)
