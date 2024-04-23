import copy

import pandas as pd
from agenttobenamed import AgentTBN


def cut_before_solve_function(code_str: str):
    # Split the input string into lines
    lines = code_str.split('\n')

    # Initialize a variable to store the position of the "def solve(df):" line
    start_pos = -1

    # Loop through the lines to find the index of the "def solve(df):" line
    for i, line in enumerate(lines):
        if line.strip().startswith('def solve(df):'):
            start_pos = i + 1
            break

    # If "def solve(df):" was found, slice the list from the next line onwards
    # Join the sliced list back into a string and return
    if start_pos != -1:
        return '\n'.join(lines[start_pos:])
    else:
        # If "def solve(df):" was not found, return the original string or an empty string
        return code_str

# Load the data
questions = pd.read_excel("dataset/dataset_250_with_code_from_gpt35.xlsx")

questions_new = copy.deepcopy(questions)

# run agent on each question
for i, question in questions.iterrows():
    agent = AgentTBN(f"dataset/dataset_tables/{question['table_name']}",
                     max_debug_times=0,
                     prompt_strategy="coder_only_functions",
                     coder_model="codellama/CodeLlama-7b-Instruct-hf", # Has no infilling mode, best for completion
                     add_column_description=True,
                     n_column_samples=2,
                     head_number=2,
                     tagging_strategy="openai", #"openai", "zero_shot_classification"
                     coder_quantization_bits=None, # for codellamas from HF: 4, 8
                     collect_inputs_for_completion=True,
                     query_type="general"
                     )

    if question["tagged_query_type"] == "plot" or question["successful_code_execution"] != True:
        questions_new.drop(i, inplace=True)
        print(f"Skipping question {i} because it's a plot or the code execution was not successful.")
        continue

    _, input_prompt = agent.answer_query(question["user_query"],
                                              show_plot=False,
                                              save_plot_path=f"plots/{i}.png",
                                              )
    questions_new.at[i, "input_prompt"] = input_prompt

    cut_code = cut_before_solve_function(question['final_generated_code']).split("\n")
    cut_code = [line for line in cut_code if "print(solve(df))" not in line]
    cut_code = "\n".join(cut_code)
    cut_code = cut_code + "\n```"

    questions_new.at[i, "completed_code"] = "```python\ndef solve(df):\n" + cut_code

questions_new.to_excel("dataset/dataset_less_than_250_without_plots_for_instruct.xlsx", index=False)


# Forgot to add backticks
# df_tmp = pd.read_excel("dataset/dataset_less_than_250_without_plots_for_completion.xlsx")
# for i, row in df_tmp.iterrows():
#     df_tmp.at[i, "completed_code"] = row["completed_code"] + "\n```"
#
# df_tmp.to_excel("dataset/dataset_less_than_250_without_plots_for_completion_backticks.xlsx", index=False)