"""
Push three datasets to the huggingface hub
"""

import datasets
import pandas as pd


# Load the datasets from "dataset/pandas_corpus/filtered_DS-1000.xlsx", "dataset/pandas_corpus/filtered_OSS-Instruct.xlsx", and "dataset/pandas_corpus/formatted_pandas_api_examples.jsonl"
df = pd.read_excel("dataset/pandas_corpus/filtered_DS-1000.xlsx")
df2 = pd.read_excel("dataset/pandas_corpus/filtered_OSS-Instruct.xlsx")
df3 = pd.read_json("dataset/pandas_corpus/formatted_pandas_api_examples.jsonl", lines=True)

# Convert the dataframes to datasets
dataset = datasets.Dataset.from_pandas(df)
dataset2 = datasets.Dataset.from_pandas(df2)
dataset3 = datasets.Dataset.from_pandas(df3)

# Push the datasets to the huggingface hub as splits of the dataset "pandas_corpora"

repo = "poludmik/pandas_documentation"
dataset3.push_to_hub(repo, split="pandas_documentation_examples")
dataset.push_to_hub(repo, split="DS1000")
dataset2.push_to_hub(repo, split="OSSInstruct")





