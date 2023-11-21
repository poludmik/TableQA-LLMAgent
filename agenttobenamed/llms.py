from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from openai import OpenAI as OpenAI_assistants

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
    DEBUGGER=2


class LLM:

    def __init__(self, model="gpt-3.5-turbo-1106", use_assistants=True):
        self.model = model
        if use_assistants:
            self._call_openai_llm = self._get_response_from_assistant
            self.client = OpenAI_assistants()
            self.my_assistant = self.client.beta.assistants.create(
                # instructions=init_instructions,
                name="Helpfull Data Analyst",
                model=model
            )
        else:
            self._call_openai_llm = self._get_response_from_base_gpt

    def get_prompt_from_router(self, user_query, df, save_plot_name):
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
        llm = ChatOpenAI(model=self.model).bind(
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

    def _get_response_from_base_gpt(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING LANGCHAIN.LLM{RESET}: {str(role)}")
        chat_model = ChatOpenAI()
        messages = [HumanMessage(content=prompt_in)]
        res = chat_model.invoke(messages).content
        print(f"{BLUE}LANGCHAIN.LLM MESSAGE{RESET}: {res}")
        return res

    def _get_response_from_assistant(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING OPENAI ASSISTANT{RESET}: {str(role)}")

        # init_instructions = "You are an AI data analyst and your job is to assist the user with simple data analysis."
        # if role == Role.CODER:
        #     init_instructions += " You generate clear and simple code following the given instructions."

        thread = self.client.beta.threads.create()

        # Add a message to a thread
        # message = client.beta.threads.messages.create(
        #     thread_id=thread.id,
        #     role="user",
        #     content="",
        # )

        # Run the assistant (start it)
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.my_assistant.id,
            instructions=prompt_in,
        )

        # messages = client.beta.threads.messages.list(thread_id=thread.id)
        # print(f"{BLUE}USER MESSAGE{RESET}: {messages.data[-1].content[0].text.value}")

        # Get the status of a Run
        run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        count = 0
        while run.status != "completed" and run.status != "failed":
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(f"{YELLOW}    RUN STATUS{RESET}: {run.status}")
            time.sleep(1.5)
            count += 1

        if run.status == "failed":
            return None

        # print(f"{BLUE}RUN OBJECT{RESET}: {run}")
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        print(f"{BLUE}ASSISTANT MESSAGE{RESET}: {messages.data[0].content[0].text.value}")
        return messages.data[0].content[0].text.value

    def plan_steps_with_gpt(self, user_query,
                            df,
                            save_plot_name):

        selected_prompt = self.get_prompt_from_router(user_query, df, save_plot_name).text
        print(selected_prompt)
        return self._call_openai_llm(selected_prompt, role=Role.PLANNER)

    def generate_code_with_gpt(self, user_query,
                               df,
                               plan):

        prompt = Prompts.generate_code.format(input=user_query, df_head=df.head(1), plan=plan)
        return self._call_openai_llm(prompt, role=Role.CODER)

    def fix_generated_code(self, code_to_be_fixed, error_message):
        prompt = Prompts.fix_code.format(code=code_to_be_fixed, error=error_message)
        return self._call_openai_llm(prompt, Role.DEBUGGER)
