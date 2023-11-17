import pandas as pd
from pathlib import Path

import random

from .llms import LLM
from .code_manipulation import Code


class AgentTBN:
    def __init__(self, csv_path: str, max_debug_times: int = 2):
        self.filename = Path(csv_path).name
        self.df = pd.read_csv(csv_path)
        self.max_debug_times = max_debug_times
        print('damn!')

    def answer_query(self, query: str):
        possible_plotname = "plots/" + self.filename[:-4] + str(random.randint(10, 99)) + ".png"
        plan = LLM.plan_steps_with_gpt(query, self.df, save_plot_name=possible_plotname)

#         plan = """1. Sort
# 2. Succeed
# 3. Win
# """

        generated_code = LLM.generate_code_with_gpt(query, self.df, plan, model="gpt-4-1106-preview")
        code_to_execute = Code.extract_code(generated_code, provider='local') # 'local' removes the definition of a new df if there is one

#         code_to_execute = """# import necessary libraries
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. Sort the DataFrame in descending order based on the 'happiness_index' column
# df = df.sort_values('happiness_index', ascending=False)
#
# # 2. Extract the top 10 rows from the sorted DataFrame to get the largest happiness indices
# top_10_largest = df.head(10)
#
# # 3. Create a pie plot using the extracted top 10 rows
# plt.pie(top_10_largest['happiness_index'], labels=top_10_largest['country'], autopct='%1.1f%%')
#
# # 4. Set the aspect ratio of the pie plot to be equal to make it a circle
# plt.axis('equal')
#
# # 5. Add a title to the pie plot
# plt.title('Top 10 Largest Happiness Indices by Country')
#
# # 6. Save the pie plot
# plt.savefig('plots/test_CSV_file_gdp81.png')
#
# # Output the result
# print("Pie plot of top 10 largest happiness indices has been saved to 'plots/test_CSV_file_gdp81.png'")
# """

        res, exception = Code.execute_generated_code(code_to_execute, self.df)

        count = 0
        while res == "ERROR" and count < self.max_debug_times:
            regenerated_code = LLM.fix_generated_code(code_to_execute, exception)
            code_to_execute = Code.extract_code(regenerated_code, provider='local')
            res, exception = Code.execute_generated_code(code_to_execute, self.df)
            count += 1

        return res


