import os

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torch

from .llms import LLM
from .code_manipulation import Code
from .logger import *

class AgentTBN:
    def __init__(self, table_file_path: str,
                 max_debug_times: int = 2,
                 gpt_model="gpt-3.5-turbo-1106",
                 coder_model="gpt-3.5-turbo-1106",
                 quantization_bits=None, # for local coding LLMs
                 adapter_path="",
                 head_number=2,
                 prompt_strategy="simple",
                 tagging_strategy="openai",
                 add_column_description=False,
                 use_assistants_api=False,
                 query_type=None, # fixes query type for all upcoming queries
                 collect_inputs_for_completion=False,
                 ):
        self.filename = Path(table_file_path).name
        self.head_number = head_number
        self.prompt_strategy = prompt_strategy
        self.add_column_description = add_column_description
        self._table_file_path = table_file_path
        self._df = None

        self.gpt_model = gpt_model
        self.coder_model = coder_model
        self.adapter_path = adapter_path
        self.quantization_bits = quantization_bits
        self.max_debug_times = max_debug_times

        # for skipping the reasoning part
        self._user_set_plan = None
        self._tagged_query_type = None
        self._prompt_user_for_planner = None

        self.user_has_fixed_query_type = True if query_type else False
        self.fixed_query_type = query_type
        assert query_type in [None, "plot", "general"], "query_type must be either None, 'plot' or 'general'."

        self.provider = "openai"
        if not self.coder_model.startswith("gpt"):
            self.provider = "local"

        self.collect_inputs_for_completion = collect_inputs_for_completion

        self.use_assistants_api = use_assistants_api
        assert not (self.use_assistants_api and self.prompt_strategy == "coder_only_simple"), "Both use_assistants_api and coder_only_simple cannot be True at the same time."

        self.llm_calls = LLM(use_assistants_api=use_assistants_api,
                             model=self.gpt_model,
                             head_number=self.head_number,
                             prompt_strategy=self.prompt_strategy,
                             add_column_description=self.add_column_description
                             )

        assert tagging_strategy in ["openai", "zero_shot_classification"], "Tagging strategy must be either 'openai' or 'zero_shot_classification'."
        self.tag = self.llm_calls.tag_query_type if tagging_strategy == "openai" else self.llm_calls.tagging_by_zero_shot_classification

        pd.set_option('display.max_columns', None) # So that df.head(1) did not truncate the printed table
        pd.set_option('display.expand_frame_repr', False) # So that did not insert new lines while printing the df
        # print('damn!')

    def delete_local_llm(self):
        self.llm_calls = None
        self.llm_calls = LLM(use_assistants_api=self.use_assistants_api,
                             model=self.gpt_model,
                             head_number=self.head_number,
                             prompt_strategy=self.prompt_strategy,
                             add_column_description=self.add_column_description
                             )
        print(f"{RED}Local LLM object has been deleted and reinitialized.{RESET}")

    @property
    def df(self):  # Lazy loading, when df is first accessed.
        if self._df is None:
            print(f"DataFrame from {GREEN}{self.filename}{RESET} is being loaded...")
            self._load_df_from_file(self._table_file_path)
        return self._df

    def _load_df_from_file(self, file_path):
        """
        Internal method to load a DataFrame from a given file path.
        Supports CSV and Excel files.
        """
        if file_path.endswith('.csv'):
            self._df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self._df = pd.read_excel(file_path)
        else:
            raise Exception("Only CSVs and XLSX files are currently supported.")

    def load_new_df(self, new_file_path: str):
        """
        Loads a new DataFrame from a given file path and updates the instance accordingly.
        """
        self._table_file_path = new_file_path
        self.filename = Path(new_file_path).name
        self._df = None # For lazy loader to load later

    def skip_reasoning_part(self, plan: str, tagged_query_type: str, prompt_user_for_planner: str):
        self._user_set_plan = plan
        self._tagged_query_type = tagged_query_type
        self._prompt_user_for_planner = prompt_user_for_planner
        if isinstance(self._user_set_plan, str) != isinstance(self._tagged_query_type, str):
            raise Exception("Both plan and tagged_query_type must be either None or a string.")

    def _reset_skip_reasoning_part(self):
        self._user_set_plan = None
        self._tagged_query_type = None
        self._prompt_user_for_planner = None

    def answer_query(self, query: str, show_plot=False, save_plot_path=None):
        try:
            return self._answer_query(query, show_plot, save_plot_path)
        except Exception as e:
            print(f"Error in answer_query(): {RED}{e}{RESET}")
            return None, None

    def _answer_query(self, query: str, show_plot=False, save_plot_path=None):
        """
        Additionally returns a dictionary with info:
            - Which prompts were used and where,
            - Generated code,
            - Number of error corrections etc.
        """
        details = {}

        possible_plotname = None
        if not show_plot:  # No need to plt.show()
            if save_plot_path is None:  # Save plot to a random filepath
                possible_plotname = "plots/" + os.path.splitext(os.path.basename(self.filename))[0] + str(
                    random.randint(10, 99)) + ".png"
            else:  # Save plot to a provided filepath
                possible_plotname = save_plot_path

        if self.use_assistants_api:
            if show_plot:
                print(f"{RED}The show_plot parameter is not supported for answer_query() with the use_assistants_api parameter enabled!{RESET}")
            text_answer = self.llm_calls.pure_openai_assistant_answer(self._table_file_path, query, possible_plotname)
            return text_answer, details

        if self.user_has_fixed_query_type: # if not set by the user
            self._tagged_query_type = self.fixed_query_type
        else:
            self._tagged_query_type = self.tag(query)

        plan = self._user_set_plan
        if self._user_set_plan is None and not self.prompt_strategy.startswith("coder_only"): # Not skipping the reasoning part
            plan, self._prompt_user_for_planner = self.llm_calls.plan_steps_with_gpt(query, self.df, save_plot_name=possible_plotname, query_type=self._tagged_query_type)

        generated_code, coder_prompt = self.llm_calls.generate_code(query, self.df, plan,
                                                               show_plot=show_plot,
                                                               tagged_query_type=self._tagged_query_type,
                                                               llm=self.coder_model,
                                                               quantization_bits=self.quantization_bits,
                                                               adapter_path=self.adapter_path,
                                                               save_plot_name=possible_plotname, # for the "coder_only" prompt strategies
                                                               collect_input_prompts=self.collect_inputs_for_completion
                                                               )

        if self.collect_inputs_for_completion:
            return "collecting input prompts", coder_prompt

        code_to_execute = Code.extract_code(generated_code, provider=self.provider, show_plot=show_plot, model_name=self.coder_model, prompt_strategy=self.prompt_strategy)  # 'local' removes the definition of a new df if there is one
        details["first_generated_code"] = code_to_execute

        if "import pandas as pd" not in code_to_execute or "import matplotlib.pyplot as plt" not in code_to_execute:
            code_to_execute = Code.prepend_imports(code_to_execute)

        if self.prompt_strategy.endswith("functions"):
            code_to_execute = Code.append_result_storage(code_to_execute)

        code_to_execute = Code.format_to_pep8(code_to_execute)

        print(f"{YELLOW}>>>>>>> Formatted code:{RESET}\n{code_to_execute}")

        res, exception = Code.execute_generated_code(code_to_execute, self.df, tagged_query_type=self._tagged_query_type)
        debug_prompt = ""

        count = 0
        errors = []

        res = "ERROR" if exception == "empty exec()" else res

        while res == "ERROR" and count < self.max_debug_times:
            errors.append(exception)
            regenerated_code, debug_prompt = self.llm_calls.fix_generated_code(self.df, code_to_execute, exception, query)
            code_to_execute = Code.extract_code(regenerated_code, provider=self.provider)
            res, exception = Code.execute_generated_code(code_to_execute, self.df, self._tagged_query_type)
            count += 1
        errors = errors + [exception] if res == "ERROR" or not code_to_execute.strip() else []

        if res == "" and self._tagged_query_type == "general":
            print(f"{RED}Empty output from exec() with the text-intended answer!{RESET}")

        # to remove outputs of the previous plot, works with show_plot=True, because plt.show() waits for user to close the window
        plt.clf()
        plt.cla()
        plt.close()

        details["plan"] = plan
        details["coder_prompt"] = coder_prompt
        details["prompt_user_for_planner"] = self._prompt_user_for_planner
        details["tagged_query_type"] = self._tagged_query_type
        details["count_of_fixing_errors"] = str(count)
        details["final_generated_code"] = code_to_execute
        details["last_debug_prompt"] = debug_prompt
        details["successful_code_execution"] = "True" if res != "ERROR" else "False"
        details["result_repl_stdout"] = res
        details["plot_filename"] = possible_plotname if self._tagged_query_type == "plot" else ""
        details["code_errors"] = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(errors)])

        ret_value = res
        if res == "":
            if self._tagged_query_type == "general":
                ret_value = "Empty output from the exec() function for the text-intended answer."
            elif self._tagged_query_type == "plot":
                ret_value = possible_plotname

        self._reset_skip_reasoning_part()

        return ret_value, details
