import pandas as pd
import json

df = pd.read_excel('dataset_82.xlsx')
formatted_records = []

for index, row in df.iterrows():
    record = {
        "coder_prompt": row["coder_prompt"],
        "plan": row["plan"],
        "generated_code": row["final_generated_code"],
        "query_id": index
    }
    formatted_records.append(record)

with open('dataset_82_coder.jsonl', 'w') as f:
    for record in formatted_records:
        json.dump(record, f)
        f.write('\n')
