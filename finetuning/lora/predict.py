from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch

seed = 1337

model_name = "codellama/CodeLlama-7b-Python-hf"
# model_name = "facebook/opt-350m"

quantization_config = BitsAndBytesConfig(load_in_4bit=False)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=quantization_config,
                                             device_map="auto"
                                             )

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_model_id = "finetuning/lora/cp/codellama_python/final"
peft_model = PeftModel.from_pretrained(model, peft_model_id, offload_folder="finetuning/lora/offload/codellama_python")
# peft_model = model

repo = "poludmik/code_completion_for_data_analysis"
dataset = load_dataset(repo, split="train")  # everything is in the train split on HF
split_ratio = 0.5  # 10% for validation
num_validation_samples = int(len(dataset) * split_ratio)
split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)
small_dataset = split_dataset['test'].select(range(20))
train_dataset = split_dataset['train']
validation_dataset = small_dataset


input_prompt = '''```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def solve(df: pd.DataFrame):
    """ Function to solve the user query: 'Select rows where temperature is greater than 30 and starting SoC is lower than 25.'.

    DataFrame `df` is fixed. The resut of `print(df.head(2))` is:
           ChargingCycleId CarType  SoC_Start (pc)  SoC_End (pc)  Temperature (C)  Current (A)  Temperature (C) - SoC_Start (pc)
    0                1  Taycan              21            24               12          160                                -9
    1                2  Taycan              26            52               22           10                                -4

    Here is also a list of column names along with the first sample values for your convenience (each column is represented as a separate json object within a list):
    [{'column_name': 'ChargingCycleId', 'sample_values': [1, 2]}, {'column_name': 'CarType', 'sample_values': ['Taycan', 'Taycan']}, {'column_name': 'SoC_Start (pc)', 'sample_values': [21, 26]}, {'column_name': 'SoC_End (pc)', 'sample_values': [24, 52]}, {'column_name': 'Temperature (C)', 'sample_values': [12, 22]}, {'column_name': 'Current (A)', 'sample_values': [160, 10]}, {'column_name': 'Temperature (C) - SoC_Start (pc)', 'sample_values': [-9, -4]}]


    Args:
        df: pandas DataFrame

    Returns:
        Variable containing the answer to the task 'Select rows where temperature is greater than 30 and starting SoC is lower than 25.' (typed e.g. float, DataFrame, list, string, dict, etc.).
    """
    # Code:\n    '''

# input_prompt = train_dataset[2]["input"] + "# Code:\n    "
# input_prompt = """\"I Am Curious: Yellow\" is a risible and pretentious steaming"""
input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
with torch.cuda.amp.autocast():
    generation_output = peft_model.generate(
        input_ids=input_tokens,
        max_new_tokens=300,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=1e-9,
        # repetition_penalty=1.1, # needs to be > 1?
        # num_return_sequences=1, # no effect
        eos_token_id=tokenizer.eos_token_id,
      )

op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(">>>>"+op+"<<<<")

