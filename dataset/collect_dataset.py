from agenttobenamed import AgentTBN
import time
import os
import pandas as pd
import pprint

start_time = time.time()

folder_path = "dataset/dataset_tables/"

llm_model = "gpt-3.5-turbo-1106"
save_dataset_path = f"dataset/dataset_changed_{llm_model}".replace('.', '_')+".xlsx"
print(save_dataset_path)

dataset_df = pd.read_excel("dataset/dataset_clean.xlsx").astype(str)
# print(dataset_df)

q_idx = [2,3,4,5,6]
for index, row in dataset_df.iterrows():
    if index not in q_idx:
        continue
    if type(row["user_query"]) != str:
        continue

    # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
    agent = AgentTBN(folder_path + row["table_name"], max_debug_times=2, gpt_model=llm_model)
    save_plot_to = "dataset/produced_plots/" + os.path.splitext(row["table_name"])[0] + "_idx" + str(index) + ".png"
    result, details = agent.answer_query(row["user_query"], save_plot_path=save_plot_to)

    for key in details.keys():
        dataset_df.loc[index, key] = details[key]

dataset_df.to_excel(save_dataset_path)

# print(pprint.pformat(details))
print(f"Elapsed time: {time.time() - start_time} seconds")






