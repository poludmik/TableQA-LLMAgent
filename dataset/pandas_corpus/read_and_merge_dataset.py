import pandas as pd
import datasets

repo = "poludmik/pandas_documentation"
dataset1 = datasets.load_dataset(repo, split="pandas_documentation_examples")
dataset2 = datasets.load_dataset(repo, split="DS1000")
dataset3 = datasets.load_dataset(repo, split="OSSInstruct")

# merge the datasets
dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3])

# drop the 'text' column
dataset = dataset.remove_columns("text")

print(dataset[1000]['code'])
