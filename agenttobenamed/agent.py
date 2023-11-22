import pandas as pd
from pathlib import Path

import random

from .llms import LLM
from .code_manipulation import Code


class AgentTBN:
    def __init__(self, csv_path: str, max_debug_times: int = 2, gpt_model="gpt-3.5-turbo-1106"):
        self.filename = Path(csv_path).name
        self.df = pd.read_csv(csv_path)
        self.gpt_model = gpt_model
        self.max_debug_times = max_debug_times
        pd.set_option('display.max_columns', None) # So that df.head(1) did not truncate the printed table
        # print('damn!')

    def answer_query(self, query: str, show_plot=False):
        llm_cals = LLM(use_assistants_api=False, model=self.gpt_model)

        possible_plotname = None
        if not show_plot:
            possible_plotname = "plots/" + self.filename[:-4] + str(random.randint(10, 99)) + ".png"

        plan, _ = llm_cals.plan_steps_with_gpt(query, self.df, save_plot_name=possible_plotname)

        generated_code, _ = llm_cals.generate_code_with_gpt(query, self.df, plan)
        code_to_execute = Code.extract_code(generated_code, provider='local', show_plot=show_plot) # 'local' removes the definition of a new df if there is one

        res, exception = Code.execute_generated_code(code_to_execute, self.df)

        count = 0
        while res == "ERROR" and count < self.max_debug_times:
            regenerated_code, _ = llm_cals.fix_generated_code(self.df, code_to_execute, exception)
            code_to_execute = Code.extract_code(regenerated_code, provider='local')
            res, exception = Code.execute_generated_code(code_to_execute, self.df)
            count += 1

        return possible_plotname if res == "" else res, possible_plotname

    def answer_query_with_details(self, query: str):
        """
        Returns a dictionary with info:
            - Which prompts were used and where,
            - Generated code,
            - Number of error corrections etc.
        """
        details = {}

        llm_cals = LLM(use_assistants_api=False, model=self.gpt_model)
        possible_plotname = "plots/" + self.filename[:-4] + str(random.randint(10, 99)) + ".png"

        plan, planner_prompt = llm_cals.plan_steps_with_gpt(query, self.df, save_plot_name=possible_plotname)
        details["prompt_user_for_planner"] = planner_prompt[0]
        details["tagged_query_type"] = planner_prompt[1]

        generated_code, coder_prompt = llm_cals.generate_code_with_gpt(query, self.df, plan)
        code_to_execute = Code.extract_code(generated_code, provider='local')  # 'local' removes the definition of a new df if there is one
        details["steps_for_codegen"] = coder_prompt
        details["first_generated_code"] = code_to_execute

        res, exception = Code.execute_generated_code(code_to_execute, self.df)

        debug_prompt = ""

        count = 0
        while res == "ERROR" and count < self.max_debug_times:
            regenerated_code, debug_prompt = llm_cals.fix_generated_code(self.df, code_to_execute, exception)
            code_to_execute = Code.extract_code(regenerated_code, provider='local')
            res, exception = Code.execute_generated_code(code_to_execute, self.df)
            count += 1

        details["count_of_fixing_errors"] = str(count)
        details["final_generated_code"] = code_to_execute
        details["last_debug_prompt"] = debug_prompt
        details["successful_code_execution"] = "True" if res != "ERROR" else "False"
        details["result_repl_stdout"] = res
        details["plot_filename"] = possible_plotname if planner_prompt[1] == "plot" else ""

        return possible_plotname if res == "" else res, details
