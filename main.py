from agenttobenamed import AgentTBN
import time

start_time = time.time()

# csv_path = "csvs/test_CSV_file_gdp.csv"
csv_path = "csvs/EV_Battery_Data.csv"
agent = AgentTBN(csv_path, gpt_model="gpt-3.5-turbo-1106")  # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"


# query = "What is the maximum Temperature?"
# query = "the correlation between current and voltage"
# query = "minimal value of rate of happy"
query = "Find top 10 largest charging cycles and pieplot them. Add values. Title it 'Ch. Cycles'."
# query = "Find the correlation between gdp and happiness index."
# query = "Create a barplot of the top 5 minimal values in the happiness index column."
# query = "Approximate the gdp to happiness index trend with a line and plot it."

# query = "average depth in the 'Plot' column" # Tags as 'general' text answer!!!

result, filename = agent.answer_query(query, show_plot=True)
# print(result)

end_time = time.time()

print(f"Elapsed time: {end_time - start_time} seconds")
