import json
import requests
import shutil
from msal import ConfidentialClientApplication
import configparser
import pandas as pd


def get_credentials():

    config = configparser.ConfigParser()
    config.read("/opt/ml/processing/input/onedrive.ini")
    vars = config['fcb']
    client_id = vars['client_id']
    client_secret = vars['client_secret']
    tenant_id = vars['tenant_id']

    credentials = {
        "client_id" : client_id,
        "client_secret" : client_secret,
        "tenant_id" : tenant_id
    }

    return credentials

def get_access_token():
    credentials = get_credentials()
    
    client_id = credentials['client_id']
    client_secret = credentials['client_secret']
    tenant_id = credentials['tenant_id']

    msal_authority = f"https://login.microsoftonline.com/{tenant_id}"

    msal_scope = ["https://graph.microsoft.com/.default"]

    msal_app = ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=msal_authority,
    )

    result = msal_app.acquire_token_silent(
        scopes=msal_scope,
        account=None,
    )

    # GET ACCESS TOKEN
    if not result:
        result = msal_app.acquire_token_for_client(scopes=msal_scope)

    if "access_token" in result:
        access_token = result["access_token"]
    else:
        raise Exception("No Access Token found")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    return headers

def get_file(headers):
    site_id = "noglai.sharepoint.com"
    file_id = "01ZFTZODTIDI4L5PJDRFELUUVVPMECVW5E"
    #file_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}/content"
    file_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}/workbook/worksheets('GJ 22_23')/usedRange"

    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        print("response get")
        file_data = response.json()
        values = file_data["values"]
        #print(values)
        df = pd.DataFrame(values[1:], columns=values[0])
        print(df)
    else:
        print("error reading a file")

headers = get_access_token()
get_file(headers)



        