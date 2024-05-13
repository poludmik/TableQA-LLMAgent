"""
Script to read the dataset/problematic_queries/spec_queries.xlsx, run agent on every query and save the input coder prompt in a jsonl file along with the code from the xlsx.
"""
import datasets
import pandas as pd
from datasets import Dataset, DatasetDict

from tableqallmagent import LLMAgent

folder_path = "dataset/dataset_tables/"
dataset_df = pd.read_excel("dataset/problematic_queries/spec_queries.xlsx")

new_df = pd.DataFrame(columns=["input", "output"])

for index, row in dataset_df.iterrows():
    agent = LLMAgent(folder_path + row["table_name"],
                     max_debug_times=0,
                     gpt_model="gpt-3.5-turbo-1106",
                     head_number=2,
                     add_column_description=True,
                     n_column_samples=2,
                     prompt_strategy="coder_only_completion_functions",
                     query_type="general",
                     coder_model="gpt-3.5-turbo-1106",
                     coder_quantization_bits=None,
                     coder_adapter_path="",
                     collect_inputs_for_completion=True,
                     )

    result, details = agent.answer_query(row["user_query"])

    # add new row to the DataFrame
    new_row = {"input": details[:-4], "output": row["code"] + "\n```"}
    new_df = new_df._append(new_row, ignore_index=True)

new_df.to_json("dataset/problematic_queries/spec_queries.jsonl", orient="records", lines=True)

# Add a new split to the HF dataset on poludmik/code_completion_for_data_analysis. Read the existing train split and append it with the new data.
prev_dataset = datasets.load_dataset("poludmik/code_completion_for_data_analysis")['train']
dataset = Dataset.from_pandas(new_df)

train_upd = datasets.concatenate_datasets([prev_dataset, dataset])

ds = DatasetDict({"train": prev_dataset, "train_upd": train_upd})

# print(ds["train_upd"][666]["input"] + ds["train_upd"][666]["output"])
# push the updated dataset to HF
# ds.push_to_hub("poludmik/code_completion_for_data_analysis")

# load to check
# dataset = datasets.load_dataset("poludmik/code_completion_for_data_analysis")
# print(dataset["train"][600]["input"] + dataset["train"][600]["output"])




