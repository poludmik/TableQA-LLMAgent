from agenttobenamed import AgentTBN
import time

start_time = time.time()

csv_path = "csvs/test_CSV_file_gdp.csv"
agent = AgentTBN(csv_path)


query = "What is the maximum gdp?"
# query = "Find top 10 largest happiness indices and pieplot them."
# query = "Find the correlation between gdp and happiness index."
# query = "Create a barplot of the top 5 minimal values in the happiness index column."
# query = "Approximate the gdp to happiness index trend with a line and plot it."
result = agent.answer_query(query)
# print(result)

end_time = time.time()

print(f"Elapsed time: {end_time - start_time} seconds")
