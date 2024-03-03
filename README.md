# A minimalistic LLM agent for Exploratory Data Analysis (EDA) using pandas library
## *Task*: given a CSV or a XLSX file, respond to user's query about this table by generating a python code and executing it.


**Query example:** 'I want to know the average gdp of countries that have happiness index greater than 5.5. I also need these countries.'

![alt text](https://github.com/poludmik/AgentToBeNamed/blob/master/README/AgentScreenshotNewLocal.png?raw=true)


## Flow parts that are used by agent:
* Tagging the query - using a LM to classify the query to "plot" or "general". Two methods are implemented, using OpenAI 
functions tagging of a Pydantic object and using a classifier LM (DeBERTa) to classify the query. This is needed to instruct the 
LLM to save the plot to a directory, so that it could be later sent as a Response via FastAPI to the user. Or if it is just a text
answer type of question, the LLM would be instructed not to do any plotting.

* Generating the plan - using a LM to generate a step-by-step plan for the coder. Inspired by [Solve and Plan (Wang et al., 2023)](https://arxiv.org/pdf/2305.04091.pdf). 
Helps smaller LLMs significantly.

* Generating the code - using a LM to generate a python code to do the data analysis. 
The code is then executed and the result is returned to the user.


## 4 different flow strategies are implemented:

* _"simple"_ - Tagging the query, generating the plan, generating the code, executing the code, returning the result to the user.
* _"simple_functions"_ - Same, but asking the CoderLLM to generate a python function instead of a main script. 
This way, the output could be better tested, as the agent mostly returns a single value (string, float, DataFrame, etc.).

![alt text](https://github.com/poludmik/AgentToBeNamed/blob/master/README/diagram_plan.png?raw=true)
* _"coder_only_simple"_ - No planning step, just generating the script and executing it.
* _"coder_only_functions"_ - Same, but asking the CoderLLM to generate a python function instead of a main script.

![alt text](https://github.com/poludmik/AgentToBeNamed/blob/master/README/diagram_coder.png?raw=true)

## Agent arguments:
* `max_debug_timees` - maximum number of debug times for the agent to run. If the agent runs out of debug times, it will return an error message to the user.
* `head_number` - number of first rows of a DataFrame to show to the LLM.
* `prompt_strategy` - flow strategy to use (described above).
* `coder_model` - model to use for the CoderLLM. Supported models are "codellama/CodeLlama-7b-Instruct-hf", "WizardLM/WizardCoder-1(3, 15)B-V1.0" and gpt models.
* `gpt_model` - model used for planning
* `add_column_description` - if True, the agent will add a json-formatted description of the columns to the prompt for the LLM. 
This ensures that the LLM knows the precise column names and accompanying values.
* `tagging_strategy` - strategy to use for tagging the query. Supported strategies are "openai" and "deberta".

## **answer_query** arguments:
* `show_plot` - if True, the agent will show the plot to the user interactively. If False, the agent will save the plot to a directory.
* `save_plot_path` - path to save the plot to. If None, the plotname will be generated.

**answer_query** method returns a text answer and a dictionary with details and outputs from every step of the flow.

## How to run the agent:
See the main.py file for an example of how to run the agent.


## Datasets and evaluation
Public datasets are in the "datasets" folder
Evaluation of the agent is in the "evaluation" folder.

* `evaluation/collect_answers.py`runs a given agent on a given dataset and collects the answers to a xlsx file. Configs are stored in `conf/` folder.
* `evaluation/evaluation.py` evaluates the answers in one xlsx to reference answers in another xlsx file either by string equality or asking a gpt model to compare the answers.
