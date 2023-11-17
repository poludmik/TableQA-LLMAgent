from dataclasses import dataclass

@dataclass
class Prompts:
    generate_steps_for_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns. Make sure to state saving the plot to '{plotname}' in the last step.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again. Here's the first two rows of 'df': 
{df_head}

Present your algorithm in up to six simple, clear English steps. 
Remember to explain steps rather than to write code.
You must output only these steps, the code generation assistant is going to follow them.

Here's an example of output for your inspiration:
1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
2. Extract the 5 rows from the sorted DataFrame.
3. Multiply each found value in the extracted dataframe by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Save the bar plot to 'plots/example_plot00.png'.
"""

    generate_steps_no_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again. Here's the first two rows of 'df': 
{df_head}

Present your algorithm with at most six simple, clear English steps. 
Remember to explain steps rather than to write code.
You must output only these steps, the code generation assistant is going to follow them.

Here's an example of output for your inspiration:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the third power.
5. Print the result.
"""
