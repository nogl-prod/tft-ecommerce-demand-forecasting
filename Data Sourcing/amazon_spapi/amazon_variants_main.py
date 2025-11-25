# external imports
from sp_api.base import Marketplaces
from sp_api.base.reportTypes import ReportType
import argparse

# internal imports
from config.settings import *
from src.database_connection_operators import *
from src.reports_api import *
from src.products_api import *
from src.listingsitems_api import *
from src.inventories_api import *
from src.productfees_api import *
from src.catalogitems_api import *
from src.utils import *
from utils import *

sys.path.append("../../")
from Support_Functions import *

pd.set_option('display.max_columns', None)

# Setup all parameters
parser = argparse.ArgumentParser(description="This script downloads / sources data from Amazon SP-API for building a products database.")

# amazon general app parameters
parser.add_argument(
    "--lwa_app_id", 
    type=str, 
    help="Amazon LWA App ID",
)

parser.add_argument(
    "--lwa_client_secret", 
    type=str, 
    help="Amazon LWA Client Secret",
)

parser.add_argument(
    "--sp_api_access_key", 
    type=str, 
    help="Selling Partner API Access Key",
)

parser.add_argument(
    "--sp_api_secret_key", 
    type=str, 
    help="Selling Partner API Secret Key",
)

parser.add_argument(
    "--sp_api_role_arn", 
    type=str, 
    help="Selling Partner API Role ARN",
)

parser.add_argument(
    "--marketplace", 
    type=str, 
    help="Marketplace",
)

parser.add_argument(
    "--start_date", 
    type=str, 
    help="Start Date",
)

# amazon client specific parameters
parser.add_argument(
    "--client_name", 
    type=str, 
    help="The name of the client. This is used for identification purposes in the database."
)

parser.add_argument(
    "--client_db_host",
    type=str, 
    help="Client Database Host",
)

parser.add_argument(
    "--client_db_port",
    type=str, 
    help="Client Database Port",
)

parser.add_argument(
    "--client_db_user",
    type=str, 
    help="Client Database User",
)

parser.add_argument(
    "--client_db_name",
    type=str, 
    help="Client Database Name",
)

parser.add_argument(
    "--client_db_password",
    type=str, 
    help="Client Database Password",
)

parser.add_argument(
    "--client_db_schema_name",
    type=str,
    default="amazon",
    help="Client Database Schema Name",
)

parser.add_argument(
    "--client_db_order_line_items_table_name",
    type=str,
    default="order_line_items",
    help="Client Database Order Line Items Table Name",
)

parser.add_argument(
    "--client_db_variants_table_name",
    type=str,
    default="variants",
    help="Client Database Order Line Items Table Name",
)

# app specific parameters
parser.add_argument(
    "--app_db_host",
    type=str, 
    help="Application Database Host",
)

parser.add_argument(
    "--app_db_port",
    type=str, 
    help="Application Database Port",
)

parser.add_argument(
    "--app_db_user",
    type=str, 
    help="Application Database User",
)

parser.add_argument(
    "--app_db_name",
    type=str,
    default="app",
    help="Application Database Name",
)

parser.add_argument(
    "--app_db_password",
    type=str, 
    help="Application Database Password",
)

parser.add_argument(
    "--app_db_schema_name",
    type=str,
    default="public",
    help="Application Database Schema Name",
)

parser.add_argument(
    "--app_db_customer_settings_table_name",
    type=str,
    default="customer_settings",
    help="Application Database Order Line Items Table Name",
)

args = parser.parse_args()

# now get client specific keys from the database
engine = create_engine('postgresql://'+args.app_db_user+":"+args.app_db_password+"@"+args.app_db_host+":5432/"+args.app_db_name, echo=False)
amazon_customer_credentials = import_data_AWSRDS(args.app_db_customer_settings_table_name, args.app_db_schema_name, engine)
amazon_customer_credentials[amazon_customer_credentials["customer_name"] == args.client_name]

start_date = args.start_date
client_name = args.client_name
refresh_token = amazon_customer_credentials[amazon_customer_credentials["customer_name"] == client_name]["amazon_refresh_token"].iloc[0]
seller_id = amazon_customer_credentials[amazon_customer_credentials["customer_name"] == client_name]["amazon_seller_id"].iloc[0]

print(client_name, start_date)

credentials=dict(
        refresh_token=refresh_token,
        lwa_app_id=args.lwa_app_id,
        lwa_client_secret=args.lwa_client_secret,
        aws_secret_key=args.sp_api_secret_key,
        aws_access_key=args.sp_api_access_key,
        role_arn=args.sp_api_role_arn,
    )

client_db_params = {
    "user": args.client_db_user,
    "password": args.client_db_password,
    "host": args.client_db_host,
    "database": client_name
}

marketplace = Marketplaces[args.marketplace] # can only take one single marketplace

# # PRODUCTS DATAFRAME

# get the products meta data
amazon_products = get_product_meta_data_basis_from_report(seller_id, marketplace, credentials, products_report_mapping)

# enrich amazon products data with listing items data
amazon_products = get_all_listing_items(amazon_products, seller_id, marketplace, credentials, listing_items_mapping, sku_identifier = "variant_sku")

# enrich amazon products data with inventory data
amazon_products = get_all_inventory_data(amazon_products, seller_id, marketplace, credentials, inventory_mapping, sku_identifier = "variant_sku", sublist_size = 50)

# enrich amazon products data with product pricing and offer data
amazon_products = get_all_product_pricing_and_offer_data(amazon_products, marketplace, credentials, products_mapping, asin_identifier = "asin", item_condition_identifier = "variant_condition")

# enrich amazon products data with product fees data
amazon_products = get_all_product_fees(amazon_products, marketplace, credentials, productfees_mapping, asin_identifier = "asin", price_identifier = "price")

# enrich amazon products data with ean data
amazon_products = enrich_dataframe_with_ean(amazon_products, marketplace, credentials)

save_to_db(amazon_products, args.client_db_schema_name, args.client_db_variants_table_name, client_db_params)