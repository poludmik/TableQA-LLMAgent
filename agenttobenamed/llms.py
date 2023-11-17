from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from openai import OpenAI

import time
from enum import Enum

from .logger import *
from .prompts import Prompts


class TopicClassifier(BaseModel):
    """Classify if the user asked for a vizualization, e.g. plot or graph, or asked for some general numerical result, e.g. finding correlation or maximum value."""

    topic: Literal["plot", "general"]
    "The topic of the user question. One of 'plot' or 'general'."


class Role(Enum):
    PLANNER=0
    CODER=1


class LLM:

    @staticmethod
    def get_prompt_from_router(user_query, df, save_plot_name, gpt_model):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        print(f"{BLUE}SELECTING A PROMPT:{RESET}")

        temlate_for_plot_planner = Prompts.generate_steps_for_plot.format(input=user_query, plotname=save_plot_name, df_head=df.head(1))
        temlate_for_math_planner = Prompts.generate_steps_no_plot.format(df_head=df.head(1), input=user_query)

        prompt_branch = RunnableBranch(
            (lambda x: x["topic"] == "plot", PromptTemplate.from_template(temlate_for_plot_planner)),
            (lambda x: x["topic"] == "general", PromptTemplate.from_template(temlate_for_math_planner)),
            PromptTemplate.from_template(temlate_for_math_planner) # default branch (must be included)
        )

        classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
        llm = ChatOpenAI(model=gpt_model).bind(
            functions=[classifier_function], function_call={"name": "TopicClassifier"}
        )
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=TopicClassifier, attr_name="topic"
        )
        classifier_chain = llm | parser

        chain = (RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
                 | prompt_branch
                 )
        # second_chain = prompt_branch
        result_prompt = chain.invoke({"input": user_query})
        return result_prompt

    @staticmethod
    def _get_response_from_assistant(prompt_in: str, model: str = "gpt-3.5-turbo-1106", role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING OPENAI ASSISTANT{RESET}: {str(role)}")

        init_instructions = "You are an AI data analyst and your job is to assist the user with simple data analysis."
        if role == Role.CODER:
            init_instructions += " You generate clear and simple code following the given instructions."

        client = OpenAI()
        my_assistant = client.beta.assistants.create(
            instructions=init_instructions,
            name="Helpfull Data Analyst",
            model=model
        )

        thread = client.beta.threads.create()

        # Add a message to a thread
        # message = client.beta.threads.messages.create(
        #     thread_id=thread.id,
        #     role="user",
        #     content="",
        # )

        # Run the assistant (start it)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=my_assistant.id,
            instructions=prompt_in,
        )

        # messages = client.beta.threads.messages.list(thread_id=thread.id)
        # print(f"{BLUE}USER MESSAGE{RESET}: {messages.data[-1].content[0].text.value}")

        # Get the status of a Run
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        count = 0
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(f"{YELLOW}RUN STATUS{RESET}: {run.status}")
            time.sleep(1.5)
            count += 1

        # print(f"{BLUE}RUN OBJECT{RESET}: {run}")
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(f"{BLUE}ASSISTANT MESSAGE{RESET}: {messages.data[0].content[0].text.value}")
        return messages.data[0].content[0].text.value

    @staticmethod
    def plan_steps_with_gpt(user_query,
                            df,
                            save_plot_name,
                            model="gpt-3.5-turbo-1106"): # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"

        selected_prompt = LLM.get_prompt_from_router(user_query, df, save_plot_name, model).text
        print(selected_prompt)
        return LLM._get_response_from_assistant(selected_prompt, model, role=Role.PLANNER)

    @staticmethod
    def generate_code_with_gpt(user_query,
                               df,
                               plan,
                               model="gpt-3.5-turbo-1106"):

        prompt = Prompts.generate_code.format(input=user_query, df_head=df.head(1), plan=plan)
        return LLM._get_response_from_assistant(prompt, model, role=Role.CODER)


