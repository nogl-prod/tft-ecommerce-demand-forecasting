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

class AmazonReport:
    def __init__(self, seller_id, report_type, marketplace, credentials, data_to_dict=None):
        """
        Initializes the AmazonReport class with the necessary credentials and parameters.

        Args:
            seller_id (str): The seller's ID.
            report_type (str): The type of report being requested.
            marketplace (str): The marketplace for the report.
            marketplace_id (str): The ID of the marketplace.
            credentials (object): Object containing the necessary Amazon SP-API credentials.
        """
        self.seller_id = seller_id
        self.report_type = report_type
        self.marketplace = marketplace
        self.marketplace_id = marketplace.marketplace_id
        self.credentials = credentials
        self.data_to_dict = data_to_dict
        self.reports_api = Reports(credentials=self.credentials, marketplace=self.marketplace)
    
    def request_report(self, data_to_dict):
        """
        Sends a request to the Amazon SP-API to generate a report.

        Args:
            data_to_dict: dictionary that defines parameters to be passed specific for each report. marketplaceId is passed automatically.

        Returns:
            dict: Response payload from the Amazon SP-API containing report details.
        """
        timer = Timer("Requesting report")
        dictionary = {
            'marketplaceIds': [self.marketplace_id]
        }
        if data_to_dict != None:
            for k in data_to_dict:
                dictionary[k] = data_to_dict.get(k)
        
        response = self.reports_api.create_report(reportType=self.report_type, data=dictionary)
        timer.end()
        print(response.payload)
        return response.payload

    def get_report_id(self, report_id):
        """
        Gets the report from the Amazon SP-API using the report ID.

        Args:
            report_id (str): The ID of the report.

        Returns:
            dict: Response payload from the Amazon SP-API containing report details.
        """
        timer = Timer("Getting report ID")
        response = self.reports_api.get_report(reportId=report_id)
        timer.end()
        print(response.payload)
        return response.payload

    def get_report_meta_data(self, report_document_id, file_storage_path):
        """
        Gets the report document from the Amazon SP-API using the report document ID.

        Args:
            report_document_id (str): The ID of the report document.

        Returns:
            dict: Response payload from the Amazon SP-API containing report document details.
        """
        timer = Timer("Getting report meta data")
        response = self.reports_api.get_report_document(report_document_id, decrypt=True, file=file_storage_path, character_code="latin-1")
        timer.end()
        print(response.payload)
        return response.payload

    def download_report_data(self, report_document_data_info):
        """
        Downloads the report data from the given URL and converts it into a pandas DataFrame.

        Args:
            report_document_data_info (dict): Information about the report document.

        Returns:
            DataFrame: Report data as a pandas DataFrame.
        """
        timer = Timer("Downloading report data")
        url = report_document_data_info['url']
        response = requests.get(url)
        report_data = response.content
        report_df = pd.read_csv(io.BytesIO(report_data), delimiter='\t', encoding='latin-1')
        timer.end()
        return report_df

    def get_report_dataframe(self, report_id=None, sleep_time=10, max_attempts=3):
            """
            Gets the report data as a pandas DataFrame.

            Args:
                sleep_time (int, optional): Time to wait before attempting to get the report. Defaults to 60 seconds.
                max_attempts (int, optional): Maximum number of attempts to try downloading the report. Defaults to 3 times.

            Returns:
                DataFrame: Report data as a pandas DataFrame.

            Raises:
                Exception: If the report is still not ready after the maximum number of attempts.
            """
            timer_dataframe = Timer("Getting report dataframe")

            # Get the reportId
            if report_id == None:
                report_id = self.request_report(self.data_to_dict)["reportId"]

            # Initialize attempts counter
            attempts = 0

            while attempts < max_attempts:
                try:
                    # Wait before attempting to get the report
                    time.sleep(sleep_time)

                    # Get the reportDocumentId
                    reportDocumentId = self.get_report_id(report_id)["reportDocumentId"]
                    
                    # If reportDocumentId is obtained without any exceptions, break the loop
                    break
                except KeyError:
                    attempts += 1
                    print(f"Report not ready. Attempt {attempts} of {max_attempts} failed. Retrying in {sleep_time} seconds...")
                    continue

            # If maximum number of attempts is reached and report is still not ready, raise an exception
            if attempts == max_attempts:
                raise Exception(f"Report not ready after {max_attempts} attempts. Please try again later.")

            # Get the report data
            report_document_meta_data = self.get_report_meta_data(reportDocumentId, file_storage_path=f"report_{report_id}.csv")

            # Get the dataframe by downloading the report
            report_df = self.download_report_data(report_document_meta_data)

            timer_dataframe.end()

            return report_df
    
def get_product_meta_data_basis_from_report(seller_id, marketplace, credentials, products_report_mapping):
    """
    Retrieves the product meta data from the Amazon report API.

    Parameters:
    seller_id (str): The seller's ID.
    marketplace (str): The marketplace ID.
    credentials: The credentials required for authentication.

    Returns:
    product_meta_data (pd.DataFrame): The normalized DataFrame containing the product meta data.
    """
    product_report = AmazonReport(seller_id, report_type = ReportType.GET_MERCHANT_LISTINGS_ALL_DATA, marketplace=marketplace, credentials=credentials)
    product_meta_data = product_report.get_report_dataframe(max_attempts = 50)
    product_meta_data.rename(columns=products_report_mapping, inplace=True)
    return product_meta_data