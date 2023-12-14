from agenttobenamed import AgentTBN
import time
import pprint

start_time = time.time()

csv_path = "csvs/Car Specs 12.xlsx"
# csv_path = "csvs/EV_Battery_Data.csv"
agent = AgentTBN(csv_path,
                 max_debug_times=2,
                 gpt_model="gpt-4-1106-preview",
                 head_number=1
                 )  # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"


# query = "What is the maximum Temperature?"
# query = "the correlation between current and voltage"
# query = "minimal value of rate of happy"
# query = "Find top 10 largest charging cycles and pieplot them. Add values. Use red color for the biggest one. Add shadow. Title it 'Ch. Cycles'."
# query = "Find the correlation between gdp and happiness index."
# query = "Create a barplot of the top 5 minimal values in the happiness index column."
query = "Pie plot top 11 accelerations"
# query = "What are 3 best accelerations?"
# query = "print the whole dataframe"
# query = "Pie plot 3 largest gdps"
# query = "Create a barplot of the 3 largest values in the Energy Throughput divided by DoD columns."
# query = "Approximate the gdp to happiness index trend with a line and plot it."

# query = "average depth in the 'Plot' column" # Tags as 'general' text answer!!!

result, details_dict = agent.answer_query(query, show_plot=False, save_plot_path="plots/kek2.png")

print("Returned result:", result)

# print("Details:", pprint.pformat(details_dict))

print(f"Elapsed time: {time.time() - start_time} seconds")
