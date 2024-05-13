import json

import tiktoken
import pandas as pd
from openai import OpenAI
import openai
from tqdm import tqdm

from tableqallmagent.coder_llms import *

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
    num_tokens = len(encoding.encode(string))
    return num_tokens

df = pd.read_excel("dataset/pandas_corpus/pandas_api_examples.xlsx")

# total = 0
# for i, row in df.iterrows():
#     method_name, examples_text = row["method_name"], row["examples"]
#     num_tokens = num_tokens_from_string(str(examples_text) + str(method_name))
#     # print(f"{method_name}: {num_tokens} tokens")
#     total += num_tokens
# print(f"Total number of tokens: {total}") # 345607 tokens


prompt = '''I scraped the pandas API documentation and extracted examples for each method.
I need to gather formatted usage examples to each method and object present in pandas documentation.
Generate a usage example for '{method_name}' in pandas. Here are also examples for it in the pandas documentation:
'{examples_text}'

You must only output an example in the format inbetween ```python and ```.
Don't generate the tests for the code, don't generate imports, and don't instantiate the objects if not necessary.
If possible, come up with random column names from any field of world knowledge.

Here are 3 examples for your reference how you should format the output:
Instance: pandas.pivot_table
Output: 
```python
table = pd.pivot_table(df, values=['Distance', 'Weight'], index=['Percentage (%)', 'Price'], aggfunc={{'Distance': "mean", 'Weight': ["min", "max", "mean"]}})
```

Instance: pandas.DataFrame.shape
Output:
```python
df.shape # outputs tuple, e.g. (3, 2)
```

Instance: pandas.core.groupby.DataFrameGroupBy.groups
Output:
```python
df.groupby(by=["a"]).groups # {{1: [0, 1], 7: [2]}} for a DataFrame df with column 'a' containing [1, 1, 7]
```
''' # 268 tokens (times 2000 is 536000 tokens)

client = OpenAI()

completions = []
found = False
for i, row in tqdm(df.iterrows()):
    if row["method_name"] == "pandas.io.formats.style.Styler.text_gradient":
        found = True
        continue
    if not found:
        continue
    method_name, examples_text = row["method_name"], row["examples"]
    formatted_prompt = prompt.format(method_name=method_name, examples_text=examples_text)
    completion = GPTCoder().query(model_name="gpt-3.5-turbo", input_text=formatted_prompt, temperature=0.1)
    print()
    print()
    print(completion)
    completions.append(completion)

    # save completions to a jsonl file (dump to json))
    with open("dataset/pandas_corpus/formatted_pandas_api_examples3.jsonl", "w") as f:
        for completion in completions:
            try:
                code_snippet = re.findall(r"```python\s*([\s\S]*?)\s*```", completion)
                extracted_code = "\n".join(code_snippet)
            except Exception as e:
                extracted_code = ""

            json.dump({"text": completion, "code": extracted_code}, f)
            f.write('\n')

