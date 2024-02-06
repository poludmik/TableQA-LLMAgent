import random
from datasets import load_dataset
import pandas as pd
import pandasql
import re


def dict_to_dataframe(table_dict):
    return pd.DataFrame(table_dict['rows'], columns=table_dict['header'])


def convert_column_types(df):
    for column in df.columns:
        # Try converting to numeric (int/float)
        backup_df = df.copy()
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # If column could not be fully converted to numeric, try converting to datetime
        if df[column].isnull().any():
            df[column] = pd.to_datetime(df[column], errors='coerce')

        if df[column].isnull().any():
            df[column] = backup_df[column]
    return df

def format_aggregate_functions(query):
    # Regular expression to find aggregate functions followed by quoted column names
    # Using [^"]+ to match any character except double quotes
    pattern = r'(MAX|MIN|COUNT|SUM|AVG)\s+"([^"]+)"'

    def replacer(match):
        function, column = match.groups()
        return f'{function}("{column}")'

    formatted_query_new = re.sub(pattern, replacer, query)

    return formatted_query_new

def is_substring_of_others(substring, str_list):
    all_parent_strings = []
    for string in str_list:
        if substring in string and substring != string:
            all_parent_strings.append(string)
    return all_parent_strings

def format_sql_query(query, df):
    all_columns = sorted(df.columns.tolist(), key=len, reverse=True)
    for col in all_columns:
        bigger_strings = is_substring_of_others(col, all_columns)
        if bigger_strings:
            hash_strings = []
            for bigger_string in bigger_strings:
                hash_string = str(random.randint(100000, 99999999))
                hash_strings.append("#" + hash_string + "#")
                query = query.replace(bigger_string, "#" + hash_string + "#")
            query = query.replace(col, f'"{col}"')
            for bigger_string, hash_string in zip(bigger_strings, hash_strings):
                query = query.replace(hash_string, bigger_string)
        else:
            query = query.replace(col, f'"{col}"')

    where_clause = re.search(r'WHERE (.+)', query, re.IGNORECASE)
    if where_clause:
        conditions = where_clause.group(1).split('AND')
        for condition in conditions:
            if '=' in condition:
                key, value = condition.split('=')
                if not value.strip().isdigit():  # if value is not a digit, add quotes
                    value = value.replace("\"", "")
                    query = query.replace(condition, f"{key}=\"{value.strip()}\"")

    query = format_aggregate_functions(query)
    # print(query)

    return query

def execute_query(query, df):
    pysqldf = lambda q: pandasql.sqldf(q, globals())
    query = query.replace("table", "df_table")
    formatted_query = format_sql_query(query, df)
    return pysqldf(formatted_query), formatted_query

def contains_non_ascii(df):
    for element in df.to_numpy().flatten():
        if isinstance(element, str):
            if any(ord(char) > 127 for char in element):
                return True
    return False

dataset = load_dataset("wikisql")

# Create a dataframe and fill it's columns: table_name, user_query, has_plot_answer, sql_commands and sql_answer
df = pd.DataFrame(columns=["table_name", "user_query", "has_plot_answer", "sql_answer"])
table_names = []
user_queries = []
has_plot_answers = []
sql_commands = []
sql_answers = []
i = -1
while len(sql_answers) < 1000:
    # print()
    i += 1
    try:
        # generate a table name
        table_name = dataset["train"][i]["table"]["name"] + ".xlsx"
        one_item = dataset["train"][i]

        # non-ascii chars in the question
        if any(ord(c) > 127 for c in one_item["question"]):
            continue

        df_table = dict_to_dataframe(one_item["table"])
        df_table = convert_column_types(df_table)

        if contains_non_ascii(df_table):
            continue

        # save a .xlsx table
        df_table.to_excel(f"dataset/wikisql/dataset_tables/{table_name}")

        # print(df_table)

        result_df, formatted_query = execute_query(one_item["sql"]["human_readable"], df_table)
        # print(result_df)

        if result_df.size == 1:
            sql_answers.append(result_df.iloc[0, 0])
        else:
            sql_answers.append(result_df.to_string(index=False))

        table_names.append(table_name)
        has_plot_answers.append("False")
        user_queries.append(one_item["question"])
        sql_commands.append(formatted_query)

    except Exception as e:
        print(dataset["train"][i]["sql"]["human_readable"])
        print(e)


# Save df to .xlsx
df["table_name"] = table_names
df["user_query"] = user_queries
df["has_plot_answer"] = has_plot_answers
df["sql_commands"] = sql_commands
df["sql_answer"] = sql_answers
df.to_excel("dataset/wikisql/clean_train_1000.xlsx", index=False)

# df = pd.read_excel("dataset/dataset_tables/robotics_sensors.xlsx")
# query = "SELECT Robot ID FROM df WHERE Error Codes = 200 OK"
# result_df, formatted_query = execute_query(query, df)
# print(result_df)

