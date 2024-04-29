import argparse
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


prompt_template = """Act as a teacher evaluating another agent's answer. The agent was given a DataFrame and asked the following question: "{user_query}".

The agent's answer was: "{answer}".

The correct answer is: "{correct_answer}".

Compare the agent's answer to the correct answer and provide a score of 0 or 1, where 0 means that the agent's answer is completely wrong and 1 means that the agent's answer is completely correct.
The answers could have any format, including a number, a string, a list, a dictionary, a DataFrame, etc. So you need to find out if the agent's answer is included in the correct answer in some way.
If the question is about a number, then compare these numbers to be equal with a precision of 0.01.
Here are some examples of correct answers, agent's answers and their scores:

- Correct answer: "228.0", agent's answer: "The maximal weight is 228", score: 1.

- Correct answer: "       Faculty Name Academic Year  Number of Undergraduate Students  Number of Graduate Students  Total Number of Students  Male Students  Female Students  International Students
0        Electrical     2022-2023                               784                          804                      1588            887              701                     357
1  Computer Science     2022-2023                               659                          854                      1513            765              748                     356
2        Mechanical     2022-2023                              1316                          649                      1965           1009              956                     362
", agent's answer: "[1588, 1513, 1965]", score: 1.

- Correct answer: "[2023-03-11, 2023-03-31, 2023-02-19]", agent's answer: "0   2023-02-19
1   2023-03-11
8   2023-03-31
Name: Last restock date, dtype: datetime64[ns]
", score: 1.

- Correct answer: "CGATCGCCGT", agent's answer: "TGCTTACGGA", score: 0.

- Correct answer: "0.85", agent's answer: "0.8494521", score: 1.

- Correct answer: "341.0", agent's answer: "    Location ID Rock Type  Depth  Density  Porosity  Magnetic Susceptibility  Permeability  Seismic Velocity
10           11   Granite  451.0   11.5      3.54                   0.9451        886.64            1337.8
", score: 0.

You must output only the score, nothing else. Example of the output format: "0" or "1".
I'll tip you 10 dollars if you get it right.
"""


def compare_with_base_answer(gpt_model: str, user_query: str, base_answer: str, agent_answer: str):
    prompt_in = prompt_template.format(user_query=user_query, answer=agent_answer, correct_answer=base_answer)
    chat_model = ChatOpenAI(model=gpt_model)
    messages = [HumanMessage(content=prompt_in)]
    res = chat_model.invoke(messages).content
    try:
        res = int(res)
    except Exception as e:
        res = 0
        print("Could not convert GPT's answer to int:", e)
    return res


def compare_answers_as_strings(base_answer: str, agent_answer: str):
    return 1 if base_answer.strip() == agent_answer.strip() else 0


def main(precise: bool, gpt_model: str, correct_answers_xlsx: str, agent_answers_xlsx: str):
    correct_answers_df = pd.read_excel(correct_answers_xlsx).astype(str)
    agent_answers_df = pd.read_excel(agent_answers_xlsx).astype(str)

    scores = []
    for index, row in correct_answers_df.iterrows():
        if row["tagged_query_type"] != "general": # or row["has_plot_answer"] == "TRUE":
            continue

        correct_answer = row["result_repl_stdout"]
        agent_answer = agent_answers_df.loc[index, "result_repl_stdout"]
        if precise:
            score = compare_answers_as_strings(correct_answer, agent_answer)
        else:
            score = compare_with_base_answer(gpt_model, row["user_query"], correct_answer, agent_answer)

        scores.append(score)
        print(f"Index: {index}, score: {score}")

    # Calculate the average score
    average_score = sum(scores) / len(scores)
    print(f"Average score on general queries: {average_score}")
    return average_score


if __name__ == "__main__":
    # Create arguments with argparse: gpt_model, correct_answers_xlsx, agent_answers_xlsx
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpt_model", type=str, default="gpt-4-0125-preview", help="Specify the GPT model to use.")
    group.add_argument("--precise", action='store_true', help="Enable precise mode. Mutually exclusive with --gpt_model.")
    parser.add_argument("--correct_answers_xlsx", type=str, default="dataset/dataset_82.xlsx")
    parser.add_argument("--agent_answers_xlsx", type=str, default="evaluation/results/filled_dataset.xlsx")
    args = parser.parse_args()

    main(precise=args.precise, gpt_model=args.gpt_model, correct_answers_xlsx=args.correct_answers_xlsx, agent_answers_xlsx=args.agent_answers_xlsx)
