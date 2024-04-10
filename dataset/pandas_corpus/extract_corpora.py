"""
Downloads datasets from huggingface and filters them
"""
import re

import pandas as pd
from datasets import load_dataset

# Load the dataset

ds_name = "xlangai/DS-1000"
# ds_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"
dataset = load_dataset(ds_name)
if ds_name == "xlangai/DS-1000":
    split = "test"
    ds_name = "DS-1000"
    column_to_remain = "reference_code"
elif ds_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
    split = "train"
    ds_name = "OSS-Instruct"
    column_to_remain = "solution"
else:
    raise ValueError("Unknown dataset")

print(dataset)



dataset[split] = dataset[split].filter(lambda x: ("df" in x[column_to_remain] or "pandas" in x[column_to_remain] or "DataFrame" in x[column_to_remain] or "dt" in x[column_to_remain]) and ("python" in x[column_to_remain] or ds_name == "DS-1000"))

print(dataset)


df = dataset[split].to_pandas()

for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(repr)

if ds_name == "OSS-Instruct":

    for i, row in df.iterrows():
        # format to contain only the code between ```python and ```
        try:
            df.at[i, column_to_remain] = re.findall(r"```python\s*([\s\S]*?)\s*```", row[column_to_remain])[0]

            # if starts with \n, remove it
            if df.at[i, column_to_remain][0] == "\\" and df.at[i, column_to_remain][1] == "n":
                df.at[i, column_to_remain] = df.at[i, column_to_remain][2:]
        except Exception as e:
            pass

if ds_name == "DS-1000":
    for i, row in df.iterrows():
        # if the string is enclosed in " or ', remove them
        if row[column_to_remain][0] == '"' and row[column_to_remain][-1] == '"':
            df.at[i, column_to_remain] = row[column_to_remain][1:-1]
        elif row[column_to_remain][0] == "'" and row[column_to_remain][-1] == "'":
            df.at[i, column_to_remain] = row[column_to_remain][1:-1]

# remove columns that are not needed
df = df[[column_to_remain]]

df.to_excel(f"dataset/pandas_corpus/filtered_{ds_name}.xlsx", index=False)

