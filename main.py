from tableqallmagent import LLMAgent
import time

start_time = time.time()

# csv_path = "dataset/dataset_tables/car_specs.xlsx"
# csv_path = "dataset/dataset_tables/random_csvs/EV_Battery_Data.csv"
csv_path = "dataset/dataset_tables/random_csvs/test_CSV_file_gdp.csv"

agent = LLMAgent(csv_path,
                 max_debug_times=1,
                 # use_assistants_api=True,
                 # gpt_model="gpt-3.5-turbo-1106", # "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0125"
                 head_number=2,
                 add_column_description=True,
                 n_column_samples=2,

                 # data_specs_dir_path="dataset/private/driving_cycles_specs.json",

                 tagging_strategy="openai",  # "openai", "zero_shot_classification"
                 # query_type="general",

                 prompt_strategy="coder_only_functions",  # "functions", "simple", "coder_only_simple", "coder_only_functions", "coder_only_infilling_functions"
                 # coder_model="claude-3-sonnet-20240229",  # "claude-2.1", "claude-3-haiku-20240307", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-1106",
                 coder_model="gpt-4o",
                 # coder_model="codellama/CodeLlama-7b-Instruct-hf",
                 # coder_model="codellama/CodeLlama-7b-Python-hf",  # Has no infilling mode, best for completion
                 #  coder_model="codellama/CodeLlama-7b-hf",
                 # coder_model="WizardLM/WizardCoder-1B-V1.0", # goes better with simple prompts, i.e. without examples
                 # coder_model="ise-uiuc/Magicoder-S-CL-7B",
                 coder_quantization_bits=4,  # for codellamas from HF: 4, 8
                 coder_adapter_path="",

                 debug_model="gpt-3.5-turbo-1106",
                 debug_quantization_bits=None,
                 debug_adapter_path="",
                 debug_strategy="completion",  # basic, completion
                 )

# query = "What is the maximum Temperature?"
# query = "the correlation between current and voltage"
# query = "minimal value of rate of happy"
# query = "Find top 10 largest charging cycles and pieplot them. Add values. Use red color for the biggest one. Add shadow. Title it 'Ch. Cycles'."
query = "Find the correlation between gdp and happiness index."
# query = "Plot happiness_index values"
# query = "Create a pie plot of the top 5 minimal values in the happiness index column in shades of blue."
# query = "I want to know the average gdp of countries that have happiness index greater than 5.5. I also need these countries."
# query = "Can you do statistics about happiness index and gdp relationship? like their max, min and mean"
# query = "filter out the countries with gdp less than 4380756541439 and find the best country in terms of gdp relative to the happy"
# query = "Pie plot top 11 accelerations"
# query = "What are 3 best accelerations relative to the length of charging cycles?"
# query = "print the whole dataframe"
# query = "Pie plot 3 largest gdps. Add values. Use red color for the biggest one. Add shadow. Title it 'GDP'."
# query = "Create a barplot of the 3 largest values in the Energy Throughput divided by DoD columns."
# query = "Approximate the gdp to happiness index trend with a line and plot it."

# query = "average depth in the 'Plot' column" # Tags as 'general' text answer!!!

# result, details_dict = agent.answer_query(query, show_plot=False, save_plot_path="plots/kek2.png")
result, details_dict = agent.answer_query(query,
                                          show_plot=True,
                                          save_plot_path="plots/testing2.png",
                                          )

print("Returned result:", result)

print("Details:", details_dict["coder_prompt"])

print(f"Elapsed time: {time.time() - start_time} seconds")

# agent.load_new_df("dataset/dataset_tables/random_csvs/EV_Battery_Data.csv")
