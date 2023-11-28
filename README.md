# A minimalistic LLM agent for Exploratory Data Analysis (EDA) using pandas library
## *Task*: given a CSV or a XLSX file, respond to user's query about this table by generating a python code and executing it.
* Using LangChain routers/tagging, a prompt type will be established: to produce a plot or to compute the numerical values.
* The GPT model breaks down the user's request into several subtasks.
* The LLM generates code based on these subtasks.
* The code is executed using the python REPL.
* If the REPL throws an error, GPT will begin debugging until resolved.
* The result is the string output of the code 'print(result)' + optionally, saved image with a plot.

**Query example:** 'Find correlation between GDP and happiness index, subtract 0.4 and multiply by 1e6.'

![alt text](https://github.com/poludmik/AgentToBeNamed/blob/master/README/AgentScreenshot.png?raw=true)
