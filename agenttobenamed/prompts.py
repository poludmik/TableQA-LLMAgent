# from dataclasses import dataclass
import pandas as pd

from .logger import *

class PromptsSimple:

    generate_steps_for_plot_save = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns. Make sure to state saving the plot to '{plotname}' in the last step. Do not include showing the plot to the user interractively; only save it to the '{plotname}'.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beggining of the 'df': 
{df_head}
{column_description}
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

    generate_steps_for_plot_show = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns. Make sure to state showing the plot in the last step.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}
Present your algorithm in up to six simple, clear English steps. 
Remember to explain steps rather than to write code.
You must output only these steps, the code generation assistant is going to follow them.

Here's an example of output for your inspiration:
1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
2. Extract the 5 rows from the sorted DataFrame.
3. Multiply each found value in the extracted dataframe by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Show the plot.
"""

    generate_steps_no_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}
Present your algorithm with at most six simple, clear English steps. 
Remember to explain steps rather than to write code.
Don't include any visualization steps like plots or charts.
You must output only these steps, the code generation assistant is going to follow them.

Here's an example of output for your inspiration:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the third power.
5. Print the result.
"""

    generate_code = """The user provided a query that you need to help achieving: '{input}'. 
You also have a list of subtasks to be accomplished using Python.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return only the python code that accomplishes the following tasks:
{plan}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

Example of the output format:
```python

```
"""

    generate_code_for_plot_save = """The user provided a query that you need to help achieving: '{input}'. 
You also have a list of subtasks to be accomplished using Python.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return only the python code that accomplishes the following tasks:
{plan}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot the way it is stated in the tasks.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

Example of the output format:
```python

```
"""

    fix_code = """You are a helpful assistant that corrects the python code that resulted in an error and returns the corrected code.

The code was designed to achieve this user request: '{input}'.
The DataFrame `df`, that we are working with has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
The execution of the following code that was provided in the previous step resulted in an error:
```python
{code}
```

The error message is: '{error}'

Return a corrected python code that fixes the error.
Always include the import statements at the top of the code, and comments and print statements where necessary.
Use the same format with backticks. Example of the output format:
```python

```
"""

    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description):
        return self.generate_steps_no_plot.format(df_head=df.head(head_number), input=user_query, column_description=column_description)

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name, column_description):
        return self.generate_steps_for_plot_save.format(input=user_query, plotname=save_plot_name, df_head=df.head(head_number), column_description=column_description)

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description):
        return self.generate_steps_for_plot_show.format(input=user_query, df_head=df.head(head_number), column_description=column_description)

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description):
        return self.generate_code.format(input=user_query, df_head=df.head(head_number), plan=plan, head_number=head_number, column_description=column_description)

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, save_plot_name=""):
        return self.generate_code_for_plot_save.format(input=user_query, df_head=df.head(head_number), plan=plan, head_number=head_number, column_description=column_description)

    def format_fix_code_prompt(self, head_number, df, user_query, code_to_be_fixed, error_message, column_description):
        return self.fix_code.format(code=code_to_be_fixed, df_head=df.head(head_number), error=error_message, input=user_query, head_number=head_number, column_description=column_description)



############### Prompts for unit testing coder LLM given a plan ###############:
class PromptsForFunctionGeneration:

    generate_steps_no_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for, and, most importantly, names of dataframe columns.

This algorithm will be later used to write a Python function which takes an existing pandas DataFrame 'df' as an argument and returns the result of the analysis. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's how the beginning of the 'df' looks: 
{df_head}
{column_description}
Present your algorithm in no more than six simple, clear English steps. 
Remember to focus on explaining the steps rather than writing code. 
Do not include any visualization steps, such as plots or charts. 
You must only output these steps; the code generation assistant will follow them to implement the solution. 
Finally, you must specify the precise value that the function should return.
Preferably, also state the Python data type of the result (e.g. float, DataFrame, list, string, dict, etc.).

Here are examples for your inspiration:

User question: 'What is the maximal voltage minus the minimal speed, but raised to the power of 3?'
Your output:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the power of 3.
5. Return the resulting number.

User question: 'Find four car ids with the largest mileage'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Mileage' column.
2. Extract the first 4 rows from the sorted DataFrame.
3. Make a list of the 'car_id' column from the extracted dataframe.
4. Return the list of car ids.
"""


    generate_steps_for_plot_save = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for and, most importantly, names of dataframe columns.
Make sure to state saving the plot to '{plotname}' in the last step. Do not include showing the plot to the user interractively; only save it to the '{plotname}'.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}
Present your algorithm in no more than six simple, clear English steps. 
Remember to focus on explaining the steps rather than writing code. 
You must only output these steps; the code generation assistant will follow them to implement the solution.

Here's an example for you:

User question: 'Create a bar plot of worst acceleration cars with voltage on the x axis and speed multiplied by 3 on the y axis'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Acceleration' column.
2. Extract the first 5 rows from the sorted DataFrame.
3. Multiply each 'Speed' value in the extracted dataframe by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Save the bar plot to 'plots/example_plot00.png'.
"""

    generate_steps_for_plot_show = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for and, most importantly, names of dataframe columns.
Make sure to state showing the plot in the last step.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}
Present your algorithm in up to six simple, clear English steps. 
Remember to explain steps rather than to write code.
You must output only these steps, the code generation assistant is going to follow them.

Here's an example for your inspiration:
User question: 'Create a bar plot of worst acceleration cars with voltage on the x axis and speed multiplied by 3 on the y axis'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Acceleration' column.
2. Extract the first 5 rows from the sorted DataFrame.
3. Multiply each 'Speed' value in the extracted dataframe by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Show the plot.
"""

    generate_code = """You are really good with Python and the pandas library. The user provided a query that you need to help achieving: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return the definition of a python function called `def solve(df):` that accomplishes the following tasks and returns the result of the analysis if needed:
{plan}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, like in the following examples:

Here are examples of the output format:

Example 1:
```python
import pandas as pd

def solve(df):
    # 1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>
    
    # 2. Extract the 5 rows from the sorted DataFrame.
    <CODE>
    
    # 3. Create a list of the 'Country' column from the extracted dataframe.
    <CODE>
    
    # 4. Return the list of countries.
    return <RESULT>
```

Example 2:
```python
import pandas as pd
import random

def solve(df):
    # 1. Find and store the minimal value in the 'Speed' column.
    <CODE>
    
    # 2. Find and store the maximal value in the 'Voltage' column.
    <CODE>
    
    # 3. Subtract the minimal speed from the maximal voltage.
    <CODE>
    
    # 4. Raise the result to the random power.
    <CODE>
    
    # 5. Return the resulting number.
    return <RESULT>
```
"""

    generate_code_for_plot_save = """You are really good with Python and the pandas library. The user provided a query that you need to help achieving: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return the definition of a python function called `def solve(df):` that accomplishes the following tasks:
{plan}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot the way it is stated in the tasks.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, like in the following examples:

Here is an example of the output format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve(df):
    # 1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>
    
    # 2. Extract the 5 rows from the sorted DataFrame.
    <CODE>
    
    # 3. Create a pie plot of the 'GDP' column of the extracted dataframe.
    <CODE>
    
    # 4. Save the pie plot to 'plots/example_plot00.png'.
    <CODE>
```
"""

    # Here, same as in SimplePrompts, but later could, for example, add column description prompts and methods would be different maybe
    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description):
        return self.generate_steps_no_plot.format(df_head=df.head(head_number), input=user_query, column_description=column_description)

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name, column_description):
        return self.generate_steps_for_plot_save.format(input=user_query, plotname=save_plot_name,
                                                        df_head=df.head(head_number), column_description=column_description)

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description):
        return self.generate_steps_for_plot_show.format(input=user_query, df_head=df.head(head_number), column_description=column_description)

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description):
        return self.generate_code.format(input=user_query, df_head=df.head(head_number), plan=plan,
                                         head_number=head_number, column_description=column_description)

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, save_plot_name=""):
        return self.generate_code_for_plot_save.format(input=user_query, df_head=df.head(head_number), plan=plan,
                                                       head_number=head_number, column_description=column_description)



class PromptsSimpleCoderOnly:

    generate_code = """The user provided a query that you need to help achieving: '{input}'.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return only the python code that accomplishes the user query.
You must include the necessary import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).

Example of the output format:
```python

```
"""

    generate_code_for_plot_save = """The user provided a query that you need to help achieving: '{input}'.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return only the python code that accomplishes the user query.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot to '{save_plot_name}' with `plt.savefig('{save_plot_name}')`.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

Example of the output format:
```python

```
"""

    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name, column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description):
        return self.generate_code.format(input=user_query, df_head=df.head(head_number), plan=plan, head_number=head_number, column_description=column_description)

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, save_plot_name=""):
        assert save_plot_name, "The save_plot_name parameter must be provided for this prompt strategy."
        return self.generate_code_for_plot_save.format(input=user_query, df_head=df.head(head_number), plan=plan, head_number=head_number, column_description=column_description, save_plot_name=save_plot_name)


class PromptsCoderOnlyForFunctionGeneration:

    generate_code = """You are really good with Python and the pandas library. The user provided a query that you need to help achieving: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return the definition of a python function called `def solve(df):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
Do not test the function with anything similar to `print(solve(df))`, only define the function, in the format of the following example:

Here are examples of the output format:

Example format:
```python
import pandas as pd

def solve(df):
    # Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>

    # Extract the 5 rows from the sorted DataFrame.
    <CODE>

    # Create a list of the 'Country' column from the extracted dataframe.
    <CODE>

    # Return the list of countries.
    return <RESULT>
```
"""

    generate_code_for_plot_save = """You are really good with Python and the pandas library. The user provided a query that you need to help achieving: '{input}'. 

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return the definition of a python function called `def solve(df):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot to '{save_plot_name}' with `plt.savefig('{save_plot_name}')`.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, in the format of the following example.

Here is an example of the output format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve(df):
    # Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>

    # Extract the 5 rows from the sorted DataFrame.
    <CODE>

    # Create a pie plot of the 'GDP' column of the extracted dataframe.
    <CODE>

    # Save the pie plot to 'plots/example_plot00.png'.
    <CODE>
```
"""

    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name,
                                                   column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description):
        return self.generate_code.format(input=user_query, df_head=df.head(head_number), plan=plan,
                                         head_number=head_number, column_description=column_description)

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description,
                                                  save_plot_name=""):
        assert save_plot_name, "The save_plot_name parameter must be provided for this prompt strategy."
        return self.generate_code_for_plot_save.format(input=user_query, df_head=df.head(head_number), plan=plan,
                                                       head_number=head_number, column_description=column_description, save_plot_name=save_plot_name)



class Prompts:
    def __init__(self, str_strategy: PromptsSimple | PromptsForFunctionGeneration,
                 head_number: int,
                 add_column_description: bool = False
                 ):

        self.head_number = head_number
        self.add_column_description = add_column_description

        if str_strategy == "functions":
            self.strategy = PromptsForFunctionGeneration()
        elif str_strategy == "simple":
            self.strategy = PromptsSimple()
        elif str_strategy == "coder_only_simple":
            self.strategy = PromptsSimpleCoderOnly()
        elif str_strategy == "coder_only_functions":
            self.strategy = PromptsCoderOnlyForFunctionGeneration()
        else:
            raise Exception(f"{RED}Unknown prompt strategy!{RESET}")

    def column_description(self, df): # Changes if we change the 'df' from outside. Creates description dynamically.
        if self.add_column_description:
            return """\nHere is also a list of column names along with the first sample values for your convenience (each column is represented as a separate json object within a list):\n""" \
                      + str([{"column_name": c, "sample_values": df[c].head(self.head_number).tolist()} for c in df.columns]) + "\n"
        return ""

    def generate_steps_no_plot_prompt(self, df, user_query):
        return self.strategy.format_generate_steps_no_plot_prompt(self.head_number, df, user_query, self.column_description(df))

    def generate_steps_for_plot_save_prompt(self, df, user_query, save_plot_name):
        return self.strategy.format_generate_steps_for_plot_save_prompt(self.head_number, df, user_query, save_plot_name, self.column_description(df))

    def generate_steps_for_plot_show_prompt(self, df, user_query):
        return self.strategy.format_generate_steps_for_plot_show_prompt(self.head_number, df, user_query, self.column_description(df))

    def generate_code_prompt(self, df, user_query, plan):
        return self.strategy.format_generate_code_prompt(self.head_number, df, user_query, plan, self.column_description(df))

    def generate_code_for_plot_save_prompt(self, df, user_query, plan, save_plot_name=""):
        return self.strategy.format_generate_code_for_plot_save_prompt(self.head_number, df, user_query, plan, self.column_description(df), save_plot_name=save_plot_name)

    def fix_code_prompt(self, df, user_query, code_to_be_fixed, error_message):
        return PromptsSimple().format_fix_code_prompt(self.head_number, df, user_query, code_to_be_fixed, error_message, self.column_description(df))
