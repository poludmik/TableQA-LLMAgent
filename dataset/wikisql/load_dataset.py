from datasets import load_dataset
import pandas as pd
from pandasql import sqldf

dataset = load_dataset("wikisql")

print(dataset["train"][0]['table'])

def dict_to_dataframe(data_dict):
    # Extract the header (column names) and the rows (data)
    columns = data_dict['header']
    rows = data_dict['rows']

    # Create the DataFrame
    df = pd.DataFrame(rows, columns=columns)

    return df

one_item = dataset["train"][0]

df_table = dict_to_dataframe(one_item["table"])

modified_sql_query = one_item["sql"]["human_readable"].replace("table", "df_table")
print(modified_sql_query)
pysqldf = lambda q: sqldf(q, globals())

# result_df = pysqldf(modified_sql_query)
result_df = pysqldf("SELECT Notes FROM df_table WHERE \"Current slogan\" = 'SOUTH AUSTRALIA'")
print(result_df)
