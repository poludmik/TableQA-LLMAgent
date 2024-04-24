from agenttobenamed.prompts import Prompts
import pandas as pd

class TestPrompts:

    def test_coder_only_completion_functions(self):

        prompts = Prompts(str_strategy="coder_only_completion_functions",
                               head_number=2,
                               add_column_description=True,
                               n_column_samples=2,
                               debug_strategy="basic",
                          )

        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
        })
        user_query = "Plot happiness_index values"
        plan = None
        column_annotation = None

        formatted = prompts.generate_code_prompt(df, user_query, plan, column_annotation)
        assert type(formatted) == str and len(formatted) > 0
