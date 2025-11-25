# external imports
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import insert
import psycopg2
import os, sys, pathlib
import importlib.util

# internal imports
src_location = pathlib.Path(__file__).absolute().parent.parent.parent.parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from Support_Functions import *

def check_schema_exists(schema_name, params):
    """
    Check if a schema exists in a PostgreSQL database.

    Parameters:
    schema_name (str): The name of the schema to check.
    params (dict): A dictionary containing the database connection parameters. 
                   It should have the keys "user", "password", "host", and "database".

    Returns:
    bool: True if the schema exists, False otherwise.
    """
    # Create the engine
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    # Create an inspector
    inspector = inspect(engine)

    # Get the list of schemas in the database
    schemas = inspector.get_schema_names()

    # Check if the schema exists
    if schema_name in schemas:
        print(f"The schema '{schema_name}' exists in the database.")
        return True
    else:
        print(f"The schema '{schema_name}' does not exist in the database.")
        return False 
    
def check_table_exists(schema_name, table_name, params):
    """
    Check if a table exists in a specific schema in a PostgreSQL database.

    Parameters:
    schema_name (str): The name of the schema to check.
    table_name (str): The name of the table to check.
    params (dict): A dictionary containing the database connection parameters. 
                   It should have the keys "user", "password", "host", and "database".

    Returns:
    bool: True if the table exists, False otherwise.
    """
    # Create the engine
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    # Create an inspector
    inspector = inspect(engine)

    # Check if the schema exists
    if schema_name in inspector.get_schema_names():
        # If the schema exists, check if the table exists within the schema
        if table_name in inspector.get_table_names(schema=schema_name):
            print(f"The table '{table_name}' exists in the schema '{schema_name}'.")
            return True

    print(f"The table '{table_name}' does not exist in the schema '{schema_name}'.")
    return False


def create_schema(schema_name, params):
    """
    Create a new schema in a PostgreSQL database.

    Parameters:
    schema_name (str): The name of the schema to create.
    params (dict): A dictionary containing the database connection parameters. 
                   It should have the keys "user", "password", "host", and "database".

    Returns:
    None
    """

    # Create the engine
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    # Create a new schema
    with engine.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        
    print(f"The schema '{schema_name}' has been created in the database.")

def save_to_db(df, schema, table, params):
    """
    Save a pandas DataFrame to a PostgreSQL database.

    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    schema (str): The name of the schema in the database where the table will be created.
    table (str): The name of the table to create in the database.
    params (dict): A dictionary containing the database connection parameters. 
                   It should have the keys "user", "password", "host", and "database".

    Returns:
    None
    """
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    df.to_sql(table, con = engine, schema=schema, if_exists='replace', index=False, chunksize=1000, method="multi")

    print(f"The dataframe has been saved to {schema}.{table}.")

def get_max_value(schema, table, column, params):
    """
    Get the maximum value of a specified column in a table of a PostgreSQL database.

    Parameters:
    schema (str): The name of the schema in the database where the table is located.
    table (str): The name of the table in the database.
    column (str): The name of the column for which to get the maximum value.
    params (dict): A dictionary containing the database connection parameters. 
                   It should have the keys "user", "password", "host", and "database".

    Returns:
    max_value: The maximum value of the specified column. The type of this will depend on the column data type.
    """
    # Create the engine
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    # Execute the query to get the maximum value
    query = f'SELECT MAX("{column}") FROM {schema}.{table}'
    max_value = pd.read_sql(query, engine).iloc[0][0]
    
    return max_value

def get_table_structure_from_database(table, schema, params):
    engine = create_engine('postgresql://'+params.get("user")+":"+params.get("password")+"@"+params.get("host")+":5432/"+params.get("database"), echo=False)

    t = Timer(schema+table)
    chunks = list()
    for chunk in pd.read_sql("SELECT * FROM "+schema+"."+table+" LIMIT 1", con=engine, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(list(chunks))
    t.end()
    return pd.DataFrame(columns=df.columns)

def map_postgresql_to_pandas_dtype(postgresql_dtype):
    dtype_mapping = {
        'bigint': 'int64',
        'integer': 'int32',
        'smallint': 'int16',
        'double precision': 'float64',
        'real': 'float32',
        'text': 'object',
        'boolean': 'bool',
        'timestamp': 'datetime64[ns]',
        'interval': 'timedelta64[ns]',
        'date': 'datetime64[ns]'
    }
    return dtype_mapping.get(postgresql_dtype, 'object')

def update_table_from_dataframe(df, table, schema, params, unique_ids):
    """
    Updates the specified table in the PostgreSQL database with the data from the provided DataFrame.

    The DataFrame is first uploaded to a temporary table in the PostgreSQL database. Then, SQL statements are executed to update the specified table from the temporary table.

    Parameters:
    df (pd.DataFrame): The DataFrame with the data to insert or update.
    database (str): The name of the PostgreSQL database.
    user (str): The username to connect to the PostgreSQL database.
    password (str): The password to connect to the PostgreSQL database.
    host (str): The host of the PostgreSQL database.
    port (str): The port of the PostgreSQL database.

    Returns:
    None
    """
    temp_table_name = "updates_"+table

    # Get the structure of the table
    structure_df = get_table_structure_from_database(table, schema, params)

    # Merge the DataFrame with the structure of the table
    df = df.reindex(columns=structure_df.columns)

    # Get the data types of the 'order_line_items' table columns
    conn = psycopg2.connect(database=params.get("database"), user=params.get("user"), password=params.get("password"), host=params.get("host"), port="5432")
    cur = conn.cursor()
    query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = '{schema}' AND table_name = '{table}';
    """
    cur.execute(query)
    column_types = cur.fetchall()
    conn.close()

    # Adjust the data types of the unique identifier columns to text
    for uid in unique_ids:
        df[uid] = df[uid].astype(str)

    # Adjust the data types of the DataFrame to match those of the corresponding database columns
    for column, dtype in column_types:
        pandas_dtype = map_postgresql_to_pandas_dtype(dtype)
        if df[column].dtype != pandas_dtype:
            try:
                if pandas_dtype.startswith('int'):
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype(pandas_dtype)
                elif pandas_dtype == 'bool':
                    df[column] = df[column].astype('bool')
                else:
                    df[column] = df[column].astype(pandas_dtype)
            except ValueError:
                print(f"Could not convert column {column} to {pandas_dtype}")

    # Upload the DataFrame to a temporary table in the PostgreSQL database
    save_to_db(df, schema, temp_table_name, params)

    # Establish a connection to the database
    conn = psycopg2.connect(database=params.get("database"), user=params.get("user"), password=params.get("password"), host=params.get("host"), port="5432")

    # Create a cursor object
    cur = conn.cursor()

    # Update unique_ids dtype to text
    for uid in unique_ids:
        sql_change_unique_id_dtypes = f"""
        ALTER TABLE {schema}.{table} ALTER COLUMN {uid} TYPE text USING {uid}::text
        """
        cur.execute(sql_change_unique_id_dtypes)
    
    # List of your columns
    columns = df.columns.tolist()

   # Generate WHERE clause based on unique_ids
    where_clause = ' AND '.join([f'{schema}.{table}.{id} = {schema}.{temp_table_name}.{id}' for id in unique_ids])

    # Your SQL statement
    # Step 2: Update existing data
    sql_update = f"""
    UPDATE {schema}.{table}
    SET {', '.join([f'{column} = {schema}.{temp_table_name}.{column}' for column in columns])}
    FROM {schema}.{temp_table_name}
    WHERE {where_clause}
    """
    cur.execute(sql_update)

    # Your SQL statement
    # Step 3: Insert new rows
    sql_insert = f"""
    INSERT INTO {schema}.{table} ({', '.join(columns)})
    SELECT {', '.join(columns)}
    FROM {schema}.{temp_table_name}
    WHERE NOT EXISTS (
        SELECT 1 FROM {schema}.{table}
        WHERE {where_clause}
    )
    """
    cur.execute(sql_insert)

    # Commit the changes
    conn.commit()

    # Step 4: Drop temporary table
    sql_drop = f"DROP TABLE {schema}.{temp_table_name}"
    cur.execute(sql_drop)

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return df