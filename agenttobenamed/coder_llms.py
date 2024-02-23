import os
import re
from os.path import exists, join, isdir
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


class CoderLLM:
    def query(self, model_name: str, input_text: str) -> str:
        pass

class GPTCoder(CoderLLM):
    def query(self, model_name: str, input_text: str) -> str:
        chat_model = ChatOpenAI(model=model_name, temperature=0)
        messages = [HumanMessage(content=input_text)]
        res = chat_model.invoke(messages).content
        return res

class CodeLlamaInstructCoder(CoderLLM):

    @staticmethod
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

    def query(self, model_name: str, input_text: str, already_loaded_model=None, adapter_path="") -> str:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if already_loaded_model is None:
            already_loaded_model = AutoModelForCausalLM.from_pretrained(
                model_name,
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
                adapter_path, _ = CodeLlamaInstructCoder.get_last_checkpoint(adapter_path)
                already_loaded_model = PeftModel.from_pretrained(already_loaded_model, adapter_path)
                already_loaded_model.eval()

        already_loaded_model.eval()
        prompt = f"<s>[INST] {input_text} [/INST]"

        max_new_tokens = 500
        temperature = 1e-9

        def generate(m, user_question, max_new_tokens_local=max_new_tokens, top_p=1, temp=temperature):
            inputs = tokenizer(user_question, return_tensors="pt").to('cuda')
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

        return generate(already_loaded_model, prompt), prompt

class WizardCoder(CoderLLM):

    def query(self, model_name: str, input_text: str) -> str:
        n_gpus = 1
        llm = LLM(model=model_name, tensor_parallel_size=n_gpus)
        print(llm)

        def evaluate_vllm(
                instruction,
                temperature=0,
                max_new_tokens=2048,
        ):
            problem_prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            prompt = problem_prompt.format(instruction=instruction)

            problem_instruction = [prompt]
            # stop_tokens = ['</s>'] # for the 34B model
            stop_tokens = ['<|endoftext|>'] # for 1B, 3B or 15B models
            sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_new_tokens,
                                             stop=stop_tokens)
            completions = llm.generate(problem_instruction, sampling_params)
            for output in completions:
                prompt = output.prompt
                print('==========================question=============================')
                print(prompt)
                generated_text = output.outputs[0].text
                print('===========================answer=============================')
                print(generated_text)
                return generated_text

        return evaluate_vllm(input_text)

