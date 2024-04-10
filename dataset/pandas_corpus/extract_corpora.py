"""
Downloads datasets from huggingface and filters them
"""

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



dataset[split] = dataset[split].filter(lambda x: "df" in x[column_to_remain] or "pandas" in x[column_to_remain] or "DataFrame" in x[column_to_remain] or "dt" in x[column_to_remain])

print(dataset)


df = dataset[split].to_pandas()

for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(repr)

# remove columns that are not needed
df = df[[column_to_remain]]

df.to_excel(f"dataset/pandas_corpus/filtered_{ds_name}.xlsx", index=False)



