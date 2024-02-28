from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser, JsonOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from openai import OpenAI

import time
from enum import Enum

from .prompts import *
from .coder_llms import *


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

    def __init__(self, model="gpt-3.5-turbo-1106",
                 use_assistants_api=False,
                 head_number=2,
                 prompt_strategy="simple",
                 add_column_description=False
                 ):
        self.model = model
        self.head_number = head_number
        self.local_coder_model = None
        self.add_column_description = add_column_description
        self.prompts = Prompts(str_strategy=prompt_strategy, head_number=head_number, add_column_description=self.add_column_description)
        if use_assistants_api:
            self._call_openai_llm = self._get_response_from_assistant
            self.client = OpenAI()
            self.my_assistant = self.client.beta.assistants.create(
                name="Data Assistant",
                instructions="""You are a data analyst who can analyse and interpret data files such as CSV and XLSX and can draw conclusions from the data. """ +\
                """If user asks for a plot or for any other sort of visualization, just return a png file with what he asked for.""", # must state to return a file
                model=self.model,
                tools=[{"type": "code_interpreter"}],
            )
            self.thread = self.client.beta.threads.create()
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
        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(df, user_query)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(df, user_query, save_plot_name)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(df, user_query)

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
        chat_model = ChatOpenAI(model=self.model, temperature=0)
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

    def plan_steps_with_gpt(self, user_query, df, save_plot_name, query_type=None):
        assert query_type is not None, "query_type must be specified (tagged before calling this function)"

        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(df, user_query)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(df, user_query, save_plot_name)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(df, user_query)

        selected_prompt = temlate_for_math_planner
        if query_type == "plot":
            selected_prompt = temlate_for_plot_planner

        # print(selected_prompt)

        return self._call_openai_llm(selected_prompt, role=Role.PLANNER), selected_prompt

    def generate_code(self,
                      user_query,
                      df,
                      plan,
                      show_plot=False,
                      tagged_query_type="general",
                      llm="gpt-3.5-turbo-1106",
                      adapter_path="",
                      save_plot_name="" # for the "coder_only" prompt strategies
                      ):
        instruction_prompt = self.prompts.generate_code_prompt(df, user_query, plan)
        if tagged_query_type == "plot" and not show_plot: # don't include plt.show() in the generated code
            instruction_prompt = self.prompts.generate_code_for_plot_save_prompt(df, user_query, plan, save_plot_name=save_plot_name)

        if llm.startswith("gpt"):
            return GPTCoder().query(llm, instruction_prompt), instruction_prompt
        elif llm == "codellama/CodeLlama-7b-Instruct-hf": # local llm
            answer, self.local_coder_model = CodeLlamaInstructCoder().query(llm,
                                                                            instruction_prompt,
                                                                            already_loaded_model=self.local_coder_model,
                                                                            adapter_path=adapter_path)
            return answer, instruction_prompt

        elif llm.startswith("WizardLM/WizardCoder-"): # under 34B
            return WizardCoder().query(llm, instruction_prompt), instruction_prompt

        elif llm == "ise-uiuc/Magicoder-S-CL-7B": # doesn't really work yet
            return CodeLlamaInstructCoder().query(llm, instruction_prompt), instruction_prompt


    def fix_generated_code(self, df, code_to_be_fixed, error_message, user_query):
        prompt = self.prompts.fix_code_prompt(df, user_query, code_to_be_fixed, error_message)
        return self._call_openai_llm(prompt, Role.DEBUGGER), prompt

    def pure_openai_assistant_answer(self, df_filename, user_query, possible_plotname=None):
        def wait_on_run(run_local, thread_local):
            while run_local.status == "queued" or run_local.status == "in_progress":
                run_local = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_local.id,
                    run_id=run_local.id,
                )
                time.sleep(0.5)
            return run_local

        def pretty_print(messages_local):
            ret_val = ""
            for m in messages_local:
                # ret_val += f"{m.role}: {m.content[0].text.value}\n"

                if len(m.content) != 0 and hasattr(m.content[0], "text"): # and has attribute text
                    ret_val += f"{m.content[0].text.value}\n"
            return ret_val

        def get_response(thread):
            return self.client.beta.threads.messages.list(thread_id=thread.id)

        def get_file_ids_from_thread(thread):
            file_ids_local = [
                file_id
                for m in get_response(thread)
                for file_id in m.file_ids
            ]
            return file_ids_local

        def write_file_to_temp_dir(file_id, output_path):
            file_data = self.client.files.content(file_id)
            file_data_bytes = file_data.read()
            with open(output_path, "wb") as f:
                f.write(file_data_bytes)

        file = self.client.files.create(
            file=open(
                df_filename,
                "rb",
            ),
            purpose="assistants",
        )

        self.my_assistant = self.client.beta.assistants.update(
            self.my_assistant.id,
            tools=[{"type": "code_interpreter"}],
            file_ids=[file.id],
        )

        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_query
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.my_assistant.id,
        )

        wait_on_run(run, self.thread)

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order="asc",
            after=message.id
        )

        # print(run.file_ids) # still only dataframe file id

        file_ids = get_file_ids_from_thread(self.thread)
        print(f"{BLUE}RETURNED FILE IDS{RESET}: {file_ids}")
        if len(file_ids) == 0:
            return pretty_print(messages)
        some_file_id = file_ids[0]
        try:
            write_file_to_temp_dir(some_file_id, possible_plotname)
        except Exception as e:
            print(f"{RED}ERROR WRITING FILE{RESET}: {e}")
            return pretty_print(messages)
        return pretty_print(messages) + " Plot saved to " + possible_plotname
