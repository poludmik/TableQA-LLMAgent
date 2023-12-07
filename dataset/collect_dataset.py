from agenttobenamed import AgentTBN
import time
import os
import pandas as pd
import pprint

start_time = time.time()

folder_path = "dataset/dataset_tables/"

dataset_df = pd.read_excel("dataset/dataset_clean.xlsx").astype(str)

print(dataset_df)

for index, row in dataset_df.iterrows():
    if index <= 6 or index >= 8:
        continue
    if type(row["user_query"]) != str:
        continue

    # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
    agent = AgentTBN(folder_path + row["table_name"], max_debug_times=3, gpt_model="gpt-4-1106-preview")
    save_plot_to = "dataset/produced_plots/" + os.path.splitext(row["table_name"])[0] + "_idx" + str(index) + ".png"
    result, details = agent.answer_query(row["user_query"], save_plot_path=save_plot_to)

    for key in details.keys():
        dataset_df.loc[index, key] = details[key]

dataset_df.to_excel("dataset/dataset_changed.xlsx")

# query = "What is the moisture level of Bamboo?"
#
# # query = "average depth in the 'Plot' column" # Tags as 'general' text answer!!!
#
# result, details = agent.answer_query_with_details(query)
#
# print(pprint.pformat(details))
#
# end_time = time.time()

print(f"Elapsed time: {time.time() - start_time} seconds")






