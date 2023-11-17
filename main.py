import pandas as pd

from agenttobenamed import AgentTBN





df = pd.read_csv("csvs/test_CSV_file_gdp.csv")
agent = AgentTBN(df)



