import json
import pandas as pd
import random
from datasets import load_dataset, Dataset

seed = 228
random.seed(seed)

data_pairs = []

skip_tab = 4

df = pd.read_excel("dataset/Text2Analysis/text2analysis_functions_with_inputs_and_outputs.xlsx")
for i, row in df.iterrows():
    data_pairs.append((row["input_prompt"], row["completed_code"]))


df = pd.read_excel("dataset/dataset_less_than_250_without_plots_for_instruct.xlsx")
for i, row in df.iterrows():
    data_pairs.append((row["input_prompt"], row["completed_code"]))


# Specify the path to your JSONL file
file_path = 'dataset/instruction_finetuning_data/instruction_inputs_outputs.jsonl'

# Open the file in write mode
with open(file_path, 'w') as file:
    # Iterate over each pair
    for input_str, output_str in data_pairs:
        # Create a dictionary for the current pair
        data = {"input": input_str, "output": output_str}
        # Convert the dictionary to a JSON string
        json_str = json.dumps(data)
        # Write the JSON string as a new line in the file
        file.write(json_str + '\n')


df = pd.read_json(file_path, lines=True)
dataset = Dataset.from_pandas(df)
# shuffle the dataset
dataset = dataset.shuffle(seed=seed) # has "input" and "output" columns

repo = "poludmik/instruct_code_for_data_analysis"
# dataset.push_to_hub(repo)

dataset = load_dataset(repo)
print(dataset["train"][500])


## Maybe I will use cross validation later
# split_ratio = 0.1  # 10% for validation
# num_validation_samples = int(len(dataset) * split_ratio)
#
# split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)
#
# # Access the training and validation sets
# train_dataset = split_dataset['train']
# validation_dataset = split_dataset['test']
#
# print("Train dataset size:", len(train_dataset))
# print("Validation dataset size:", len(validation_dataset))
#
# train_dataset.push_to_hub(repo, split="train")
# validation_dataset.push_to_hub(repo, split="validation")
