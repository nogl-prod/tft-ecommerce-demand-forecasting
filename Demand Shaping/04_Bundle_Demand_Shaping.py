import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta


matrix = pd.read_excel('/opt/ml/processing/input/' + r"Bundel-Product-Matrix-Stoertebekker.xlsx", index_col="variant_sku")

def import_data_AWSRDS(table, schema, engine):
    chunks = list()
    for chunk in pd.read_sql("SELECT * FROM "+schema+"."+table, con=engine, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(list(chunks))
    
    return df

engine = create_engine('postgresql://'+"postgres"+":"+"voids4thewin"+"@"+"voidsdb.c2wwnfcaisej.eu-central-1.rds.amazonaws.com"+":5432/"+"stoertebekker",echo=False)

df = import_data_AWSRDS(schema="forecasts",table="forecasts_28days_incl_history",engine=engine)

items = matrix.columns
bundles = matrix.index

df['isBundle'] = df['variant_sku'].isin(list(bundles))

df_bundles = df.merge(matrix,on="variant_sku")


original_columns = df_bundles.columns
target_columns = ['lineitems_quantity', 'revenue', 'NOGL_forecast_q0', 'NOGL_forecast_q1',
                'NOGL_forecast_q2', 'NOGL_forecast_q3', 'NOGL_forecast_q4',
                'NOGL_forecast_q5', 'NOGL_forecast_q6']


bundle_columns = [item + "~" + column + "~bundles" for item in items for column in target_columns]
df_bundles_new = pd.DataFrame(columns=bundle_columns)


for column in target_columns:
    for item in items:
        df_bundles_new[item + "~" + column + "~bundles"] = df_bundles[column] * df_bundles[item]

org_bundle_df = df_bundles[df_bundles["isBundle"]==True]
df_bundles = org_bundle_df.merge(df_bundles_new, left_index=True, right_index=True)

output_columns = df_bundles.iloc[:,-len(target_columns)*len(items):].columns

dates = df_bundles["daydate"].unique()


df_single = df[df["isBundle"]==False]
df_single[target_columns] = 0

for date in dates:
    for column in output_columns:
        item = column.split("~")[0]
        columnstr = column.split("~")[1]
        mask = (df_single["variant_sku"]==item) & (df_single["daydate"]==date)
        df_single.loc[mask, columnstr] = df_bundles[column].loc[df_bundles["daydate"]==date].sum()


new_cols = {col: col + "_bundle" for col in target_columns}
df_single = df_single.rename(columns=new_cols)

single_values = df[df['isBundle']==False][target_columns]
df_single = pd.concat([df_single,single_values],axis=1)

for column in target_columns:
    df_single[column+"_total"] = df_single.apply(lambda row: float(row[column] or 0.0) + float(row[column+"_bundle"] or 0.0), axis=1)

df_bundles = df.merge(matrix,on="variant_sku")

final_df = pd.merge(df_bundles, df_single, on=['variant_sku'], how='outer')
final_df = final_df.fillna(0)

engine = create_engine('postgresql://'+"postgres"+":"+"voids4thewin"+"@"+"voidsdb.c2wwnfcaisej.eu-central-1.rds.amazonaws.com"+":5432/"+"stoertebekker",echo=False)
quantity_df = import_data_AWSRDS(schema="transformed",table="shopify_products",engine=engine)

quantity_df = quantity_df[["variant_sku","variant_inventory_quantity"]]

final_df = final_df.merge(quantity_df, on="variant_sku")

engine = create_engine('postgresql://'+"postgres"+":"+"voids4thewin"+"@"+"voidsdb.c2wwnfcaisej.eu-central-1.rds.amazonaws.com"+":5432/"+"stoertebekker",echo=False)
final_df.to_sql("single_sku_attributed3", con = engine, schema="forecasts", if_exists='replace', index=False, chunksize=1000, method="multi")
