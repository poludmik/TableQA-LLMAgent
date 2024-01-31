import random
from agenttobenamed import AgentTBN
import time
import os
import pandas as pd
import pprint

start_time = time.time()

folder_path = "dataset_tables/"

llm_model = "gpt-4-1106-preview" # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
head_number = 2
max_debug_times = 2
save_dataset_path = f"dataset_changed_{llm_model}_hn-{head_number}_mdt-{max_debug_times}_7".replace('.', '_')+".xlsx"

dataset_df = pd.read_excel("dataset_changed_gpt-4-1106-preview_hn-2_mdt-2_6.xlsx").astype(str)
# print(dataset_df)

# q_idx = list(range(13, 14))
# q_idx = [77, 6, 3]
for index, row in dataset_df.iterrows():
    # if index not in q_idx:
    #     continue
    if row["user_query"] == "" or row["user_query"] == "nan": # for the last row with plot counts
        continue

    agent = AgentTBN(folder_path + row["table_name"],
                     max_debug_times=max_debug_times,
                     gpt_model=llm_model,
                     head_number=head_number
                     )

    save_plot_to = "plots/" + os.path.splitext(row["table_name"])[0] + str(index) + str(random.randint(0, 9)) + ".png"
    result, details = agent.answer_query(row["user_query"], save_plot_path=save_plot_to)

    for key in details.keys():
        dataset_df.loc[index, key] = details[key]

    dataset_df.to_excel(save_dataset_path, index=False)

dataset_df.to_excel(save_dataset_path, index=False)
print(f"Elapsed time: {time.time() - start_time} seconds")

