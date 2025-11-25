import pandas as pd
import time
import requests
import yaml
from typing import Dict
from msal import ConfidentialClientApplication
from io import BytesIO

class OneDriveAPI:
    def __init__(self, client_name: str, config_filename: str = '/opt/ml/processing/scripts/Inference/config/configOneDrive.yaml'):
        """
        Initializes an instance of the OneDriveAPI class.

        Args:
            client_id (str): The client ID of the registered Azure AD application.
            client_secret (str): The client secret of the registered Azure AD application.
            tenant_id (str): The ID of the Azure AD tenant.
            scope (list[str]): The scopes required to access the OneDrive API.
            site_id (str): The ID of the SharePoint site containing the OneDrive folder.

        Returns:
            None

        Description:
            This function initializes an instance of the OneDriveAPI class. It sets the necessary instance variables, 
            and initializes the MSAL app instance for acquiring an access token.

        """
        self.params = self.config(client_name, config_filename)
        self.client_id = self.params.get('client_id')
        self.client_secret = self.params.get('client_secret')
        self.authority = self.params.get('msal_authority')
        self.scope = self.params.get('msal_scope')
        self.site_id = self.params.get('site_id')
        self.access_token = None
        self.refresh_token = None
        self.token_expiration = None

        self.msal_app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )

        self._get_access_token()

    def config(self, section: str, filename: str) -> Dict[str, str]:
        """
        Read configuration from a YAML file.

        Args:
            section (str): The section in the config file to read.
            filename (str): The name of the YAML config file.
        
        Returns:
            dict: Dictionary containing the configuration data.
        
        Raises:
            Exception: If the specified section is not found in the file.
        """
        with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)
            return config_data
        
        
    def _get_access_token(self):
        """
        Gets an access token for accessing the OneDrive API.

        Args:
            None

        Returns:
            None

        Description:
            This function gets an access token for accessing the OneDrive API. It first tries to acquire a token
            silently, and if that fails, it acquires a new token using the client credentials.

        """
        result = self.msal_app.acquire_token_silent(
            scopes=self.scope,
            account=None
        )

        if not result:
            result = self.msal_app.acquire_token_for_client(scopes=self.scope)

        if "access_token" in result:
            self.access_token = result["access_token"]
            self.refresh_token = result.get("refresh_token")
            self.token_expiration = time.time() + result.get("expires_in")
        else:
            raise Exception("No Access Token found")
        
    def get_access_token(self):
        """
        Gets an access token for accessing the OneDrive API.

        Args:
            None

        Returns:
            access_token (str): The access token for accessing the OneDrive API.

        Description:
            This function gets an access token for accessing the OneDrive API. It checks if the token has expired,
            and if it has, it acquires a new token using the refresh token.

        """

        if self.token_expiration is None or time.time() + 300 >= self.token_expiration:
            result = self.msal_app.acquire_token_by_refresh_token(
                refresh_token=self.refresh_token,
                scopes=self.scope,
            )

            if "access_token" in result:
                self.access_token = result["access_token"]
                self.refresh_token = result.get("refresh_token")
                self.token_expiration = time.time() + result.get("expires_in")
            else:
                raise Exception("Could not refresh Access Token")

        return self.access_token

    def find_item_by_relative_path(self, relative_path):
            """
            This function retrieves the item in OneDrive corresponding to the specified relative path.

            Arguments:
            relative_path -- The relative path to the item in OneDrive.

            Output:
            Returns a JSON object representing the retrieved item.
            """
            url = f"https://graph.microsoft.com/v1.0/drive/root:/{relative_path}"
            response = requests.get(url, headers={"Authorization": f"Bearer {self.get_access_token()}"})
            response.raise_for_status()
            return response.json()
    
    def download_file_by_relative_path(self, relative_path):
        """
        This function downloads the file specified in the relative path.

        Arguments:
        relative_path (str): The relative path of the file to download

        Output:
        Returns a pandas DataFrame containing the data from the downloaded file

        Description:
        This function downloads a file from OneDrive using its relative path and returns its contents as a pandas DataFrame.
        It first calls the find_item_by_relative_path method to get the ID of the file.
        It then uses the file ID to construct the URL of the file and sends an HTTP GET request to retrieve the file information.
        If the response status code is not 200, it raises an exception.
        It extracts the file name and download URL from the response data and sends an HTTP GET request to download the file.
        If the response status code is not 200, it raises an exception.
        It then uses pandas.read_excel or pandas.read_csv method to read the file data from the response content and returns the data as a pandas DataFrame.
        """
        item = self.find_item_by_relative_path(relative_path)
        file_id = item["id"]
        file_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{file_id}"
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        
        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to retrieve file information")
            
        file_data = response.json()
        file_name = file_data["name"]
        download_url = file_data["@microsoft.graph.downloadUrl"]
        
        response = requests.get(download_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to download file")
        
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
        elif file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(response.content), engine="python", index_col=0)
        else:
            raise Exception("File format not supported")
            
        return df
    
    def load_files_from_folder(self, relative_folder_path, filename_startswith):
        """
        This function loads all files from a OneDrive folder that match a specified prefix.

        Arguments:
        relative_folder_path (str): The relative path of the folder to load files from
        filename_startswith (str): The prefix of the filenames to match

        Output:
        Returns a pandas DataFrame containing the concatenated data from all matching files.

        Description:
        This function loads all files from a OneDrive folder that start with a specified prefix and returns their data concatenated as a single pandas DataFrame.
        It first calls the find_item_by_relative_path method to get the ID of the folder.
        It then uses the folder ID to construct the URL of the folder contents and sends an HTTP GET request to retrieve the contents.
        If the response status code is not 200, it raises an exception.
        It extracts the files from the response data and reads each file into a pandas DataFrame using pandas.read_csv or pandas.read_excel method depending on the file format.
        It then appends the data to a list of data frames and concatenates all data frames into a single data frame.
        """
        item = self.find_item_by_relative_path(relative_folder_path)
        folder_id = item["id"]
        folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{folder_id}/children"
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}

        response = requests.get(folder_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to retrieve folder contents")

        files = response.json()["value"]
        df_list = []
        for file in files:
            if file["name"].startswith(filename_startswith):
                download_url = file.get("@microsoft.graph.downloadUrl")
                if download_url is None:
                    continue

                response = requests.get(download_url, headers=headers)
                if response.status_code != 200:
                    raise Exception("Failed to download file")

                if file["name"].lower().endswith(".csv"):
                    df = pd.read_csv(BytesIO(response.content), engine="python", index_col=0)
                elif file["name"].lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(BytesIO(response.content), engine="openpyxl")
                else:
                    continue

                df_list.append(df)

                print(file["name"])

        df_concatenated = pd.concat(df_list, ignore_index=True)
        return df_concatenated
