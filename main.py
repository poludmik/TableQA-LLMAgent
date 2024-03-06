from agenttobenamed import AgentTBN
import time

start_time = time.time()

# csv_path = "dataset/dataset_tables/car_specs.xlsx"
# csv_path = "dataset/dataset_tables/random_csvs/EV_Battery_Data.csv"
csv_path = "dataset/dataset_tables/random_csvs/test_CSV_file_gdp.csv"
agent = AgentTBN(csv_path,
                 max_debug_times=0,
                 # use_assistants_api=True,
                 # gpt_model="gpt-3.5-turbo-1106", # "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0125"
                 head_number=2,
                 prompt_strategy="simple", # "functions", "simple", "coder_only_simple", "coder_only_functions"
                 coder_model="gpt-3.5-turbo-1106",
                 # coder_model="codellama/CodeLlama-7b-Instruct-hf",
                 # coder_model="WizardLM/WizardCoder-1B-V1.0", # goes better with simple prompts, i.e. without examples
                 # coder_model="ise-uiuc/Magicoder-S-CL-7B",
                 add_column_description=True,
                 tagging_strategy="zero_shot_classification", #"openai", "zero_shot_classification"
                 )

# query = "What is the maximum Temperature?"
# query = "the correlation between current and voltage"
# query = "minimal value of rate of happy"
# query = "Find top 10 largest charging cycles and pieplot them. Add values. Use red color for the biggest one. Add shadow. Title it 'Ch. Cycles'."
# query = "Find the correlation between gdp and happiness index."
query = "Plot 5 worst happiness index values"
# query = "Create a pie plot of the top 5 minimal values in the happiness index column in shades of blue."
# query = "I want to know the average gdp of countries that have happiness index greater than 5.5. I also need these countries."
# query = "Can you do statistics about happiness index and gdp relationship? like their max, min and mean"
# query = "filter out the countries with gdp less than 4380756541439"
# query = "Pie plot top 11 accelerations"
# query = "What are 3 best accelerations?"
# query = "print the whole dataframe"
# query = "Pie plot 3 largest gdps. Add values. Use red color for the biggest one. Add shadow. Title it 'GDP'."
# query = "Create a barplot of the 3 largest values in the Energy Throughput divided by DoD columns."
# query = "Approximate the gdp to happiness index trend with a line and plot it."

# query = "average depth in the 'Plot' column" # Tags as 'general' text answer!!!

# result, details_dict = agent.answer_query(query, show_plot=False, save_plot_path="plots/kek2.png")
result, details_dict = agent.answer_query(query,
                                          show_plot=True,
                                          save_plot_path="plots/testing_coder_only.png",
                                          )

print("Returned result:", result)

# print("Details:", pprint.pformat(details_dict))

print(f"Elapsed time: {time.time() - start_time} seconds")
