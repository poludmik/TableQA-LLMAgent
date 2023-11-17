import random

import pandas as pd
from .llms import LLM
from pathlib import Path



class AgentTBN:
    def __init__(self, csv_path: str):
        self.filename = Path(csv_path).name
        self.df = pd.read_csv(csv_path)
        print('damn!')

    def answer_query(self, query: str):
        possible_plotname = "plots/" + self.filename[:-4] + str(random.randint(10, 99)) + ".png"
        LLM.plan_steps_with_llm(query, self.df, save_plot_name=possible_plotname)
