#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os

# Date configuration
datacollection_startdate = date(2020, 1, 1)

# Microsoft Azure AD / OneDrive credentials
# These should be set as environment variables for security
# See .env.example for required variables
client_id = os.getenv("MSAL_CLIENT_ID", "")
client_secret = os.getenv("MSAL_CLIENT_SECRET", "")
tenant_id = os.getenv("MSAL_TENANT_ID", "")

if not client_id or not client_secret or not tenant_id:
    raise ValueError(
        "Missing required environment variables for Microsoft authentication. "
        "Please set MSAL_CLIENT_ID, MSAL_CLIENT_SECRET, and MSAL_TENANT_ID. "
        "See .env.example for details."
    )

msal_authority = f"https://login.microsoftonline.com/{tenant_id}"
msal_scope = ["https://graph.microsoft.com/.default"]
site_id = os.getenv("MSAL_SITE_ID", "noglai.sharepoint.com")

# Refresh token is optional and can be set via environment variable
refresh_token = os.getenv("MSAL_REFRESH_TOKEN", "")


