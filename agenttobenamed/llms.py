# from langchain.llms import OpenAI
import re
import os
from os.path import exists, join, isdir

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser, JsonOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import time
from enum import Enum

from .logger import *
from .prompts import *


class TopicClassifier(BaseModel):
    """Classify if the user asked for a vizualization, e.g. plot or graph, or asked for some general numerical result, e.g. finding correlation or maximum value."""

    topic: Literal["plot", "general"]
    "The topic of the user question. One of 'plot' or 'general'."


class Tagging(BaseModel):
    """Tag the piece of text with particular info. Classify if the user asked for a vizualization, e.g. plot or graph, or asked for some general numerical result, e.g. finding correlation or maximum value."""
    topic: str = Field(description="Topic of user's query, must be `plot` or `general`.")


class Role(Enum):
    PLANNER=0
    CODER=1
    DEBUGGER=2


class LLM:

    def __init__(self, model="gpt-3.5-turbo-1106", use_assistants_api=True, head_number=2, prompt_strategy="simple"):
        self.model = model
        self.head_number = head_number
        self.local_coder_model = None
        self.prompts = Prompts(str_strategy=prompt_strategy)
        if use_assistants_api:
            self._call_openai_llm = self._get_response_from_assistant
            self.client = OpenAI()
            self.my_assistant = self.client.beta.assistants.create(
                # instructions=init_instructions,
                name="Helpfull Data Analyst",
                model=model
            )
        else:
            self._call_openai_llm = self._get_response_from_base_gpt

    @staticmethod
    def tag_query_type(user_query):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        model = ChatOpenAI(temperature=0)
        tagging_functions = [convert_pydantic_to_openai_function(Tagging)]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Think carefully, and then tag the text as instructed"),
            ("user", "{input}")
        ])
        model_with_functions = model.bind(
            functions=tagging_functions,
            function_call={"name": "Tagging"}
        )
        tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

        query_topic = tagging_chain.invoke({"input": user_query})["topic"] # If halts => could be a problem with parser

        print(f"{BLUE}SELECTED A PROMPT:{RESET} {YELLOW}{query_topic}{RESET}")

        return query_topic

    # Halts on chain.invoke() sometimes
    def get_prompt_from_router(self, user_query, df, save_plot_name=None):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        print(f"{BLUE}SELECTING A PROMPT:{RESET}")
        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(self.head_number, df, user_query)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(self.head_number, df, user_query, save_plot_name)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(self.head_number, df, user_query)

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
        return result_prompt.text

    def _get_response_from_base_gpt(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING LANGCHAIN.LLM{RESET}: {YELLOW}{str(role)}{RESET}")
        chat_model = ChatOpenAI(model=self.model)
        messages = [HumanMessage(content=prompt_in)]
        res = chat_model.invoke(messages).content
        print(f"{BLUE}LANGCHAIN.LLM MESSAGE{RESET}: {res}")
        return res

    def _get_response_from_assistant(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING OPENAI ASSISTANT{RESET}: {YELLOW}{str(role)}{RESET}")

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

    def plan_steps_with_gpt(self, user_query, df, save_plot_name):
        query_type = self.tag_query_type(user_query)

        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(self.head_number, df, user_query)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(self.head_number, df, user_query, save_plot_name)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(self.head_number, df, user_query)

        selected_prompt = temlate_for_math_planner
        if query_type == "plot":
            selected_prompt = temlate_for_plot_planner

        print(selected_prompt)

        return self._call_openai_llm(selected_prompt, role=Role.PLANNER), (selected_prompt, query_type)

    def generate_code(self, user_query, df, plan, show_plot=False, tagged_query_type="general", llm="gpt", adapter_path=""):
        prompt = self.prompts.generate_code_prompt(self.head_number, df, user_query, plan)
        if tagged_query_type == "plot" and not show_plot: # don't include plt.show() in the generated code
            prompt = self.prompts.generate_code_for_plot_save_prompt(self.head_number, df, user_query, plan)
        if llm == "gpt":
            return self._call_openai_llm(prompt, role=Role.CODER), prompt
        else: # local llm
            tokenizer = AutoTokenizer.from_pretrained(llm)
            if self.local_coder_model is None:
                self.local_coder_model = AutoModelForCausalLM.from_pretrained(
                    llm,
                    # torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    # load_in_4bit=True,
                    load_in_8bit=True,
                    # quantization_config=BitsAndBytesConfig(
                    #     load_in_4bit=True,
                    #     bnb_4bit_compute_dtype=torch.bfloat16,
                    #     bnb_4bit_use_double_quant=True,
                    #     bnb_4bit_quant_type='nf4',
                    # ),
                    # low_cpu_mem_usage=True
                )
                if adapter_path != "":
                    adapter_path, _ = get_last_checkpoint(adapter_path)
                    self.local_coder_model = PeftModel.from_pretrained(self.local_coder_model, adapter_path)
                    self.local_coder_model.eval()

            self.local_coder_model.eval()
            prompt = f"<s>[INST] {prompt} [/INST]"

            max_new_tokens = 500
            temperature = 1e-9

            def generate(m, user_question, max_new_tokens_local=max_new_tokens, top_p=1, temp=temperature):
                inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_local,
                    # **inputs,
                    # generation_config=GenerationConfig(
                    #     # do_sample=True,
                    # max_new_tokens=max_new_tokens,
                    #     # top_p=top_p,
                    #     # temperature=temp,
                    # )
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # print(text)

                text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
                print("Text after regex:", text)
                return text

            return generate(self.local_coder_model, prompt), prompt

    def fix_generated_code(self, df, code_to_be_fixed, error_message, user_query):
        prompt = self.prompts.fix_code_prompt(self.head_number, df, user_query, code_to_be_fixed, error_message)
        return self._call_openai_llm(prompt, Role.DEBUGGER), prompt


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed:
            print("what")
            return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    print("isdir is false")
    return None, False # first training