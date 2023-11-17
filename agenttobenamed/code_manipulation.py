import re
# import code # another way to use REPL
import io
from contextlib import redirect_stdout
from .logger import *

import pandas as pd


class Code:

    @staticmethod
    def _normalize_indentation(code_segment: str) -> str:
        # Determine the minimum indentation of non-empty lines
        lines = code_segment.strip().split('\n')
        min_indent = min(len(re.match(r'^\s*', line).group()) for line in lines if line.strip())

        # Remove the minimum indentation from each line
        return '\n'.join(line[min_indent:] for line in lines)


    # Method to clean the LLM response, and extract the code.
    @staticmethod
    def extract_code(response: str, provider: str) -> str:
        # Use re.sub to replace all occurrences of the <|im_sep|> with the ```.
        response = re.sub(re.escape("<|im_sep|>"), "```", response)

        # Define a blacklist of Python keywords and functions that are not allowed
        blacklist = ['subprocess', 'sys', 'eval', 'exec', 'socket', 'urllib',
                     'shutil', 'pickle', 'ctypes', 'multiprocessing', 'tempfile', 'glob', 'pty',
                     'commands', 'cgi', 'cgitb',
                     'xml.etree.ElementTree', 'builtins', 'subprocess', 'sys', 'eval', 'exec', 'socket',
                     'urllib', 'shutil', 'pickle',
                     'ctypes', 'multiprocessing', 'tempfile', 'glob', 'pty', 'commands', 'cgi', 'cgitb',
                     'xml.etree.ElementTree', 'builtins', 'os.system', 'os.popen', 'sys.modules',
                     '__import__', 'getattr', 'setattr', 'pickle.loads', 'execfile', 'exec', 'compile',
                     'input', 'ast.literal_eval'
                     ] # TODO: add 'os'?

        # Use a regular expression to find all code segments enclosed in triple backticks with "python"
        # code_segments = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        # Use a regular expression to find all code segments enclosed in triple backticks with or without "python"
        code_segments = re.findall(r'```(?:python\s*)?(.*?)\s*```', response, re.DOTALL)
        if not code_segments:
            code_segments = re.findall(r'\[PYTHON\](.*?)\[/PYTHON\]', response, re.DOTALL)

        # Normalize the indentation for each code segment
        normalized_code_segments = [Code._normalize_indentation(segment) for segment in code_segments]

        # Combine the normalized code segments into a single string
        code_res = '\n'.join(normalized_code_segments).lstrip()

        # Remove any instances of "df = pd.read_csv('filename.csv')" from the code
        code_res = re.sub(r"df\s*=\s*pd\.read_csv\((.*?)\)", "", code_res)

        # This is necessary for local OS models, as they are not as good as OpenAI models deriving the instruction from the promt
        if provider == "local":
            # Replace all occurrences of "data" with "df" if "data=pd." is present
            if re.search(r"data=pd\.", code_res):
                code_res = re.sub(r"\bdata\b", "df", code_res)
            # Comment out the df instantiation if it is present in the generated code
            code_res = re.sub(r"(?<![a-zA-Z0-9_-])df\s*=\s*pd\.DataFrame\((.*?)\)",
                          "# The dataframe df has already been defined", code_res)

        # Define the regular expression pattern to match the blacklist items
        pattern = r"^(.*\b(" + "|".join(blacklist) + r")\b.*)$"

        # Replace the blacklist items with comments
        code_res = re.sub(pattern, r"# not allowed \1", code_res, flags=re.MULTILINE)

        # Return the cleaned and extracted code
        return code_res.strip()

    @staticmethod
    def execute_generated_code(code_str: str, df: pd.DataFrame):
        print(f"{YELLOW}EXECUTING THE CODE{RESET}")
        with redirect_stdout(io.StringIO()) as output:
            exec(code_str, {'df': df})

        results = output.getvalue()
        output.truncate(0)
        output.seek(0)
        print(f"{YELLOW}FINISHED EXECUTING, RESULT{RESET}:\n{results}")
        return results
