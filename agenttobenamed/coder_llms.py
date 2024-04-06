import os
import re
from os.path import exists, join, isdir
import torch
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# from vllm import LLM, SamplingParams # uncomment for WizardCoder, couldn't install on cluster
from .logger import *


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
        if isdir(join(checkpoint_dir, "final")):
            return join(checkpoint_dir, "final"), True
        if isdir(checkpoint_dir):
            is_completed = exists(checkpoint_dir) and "checkpoint" in checkpoint_dir
            if is_completed:
                print("what")
                return checkpoint_dir, True  # already finished
            max_step = 0
            for filename in os.listdir(checkpoint_dir):
                if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                    max_step = max(max_step, int(filename.replace('checkpoint-', '')))
            if max_step == 0: return None, is_completed  # training started, but no checkpoint
            checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
            print(f"        Found a previous checkpoint at: {checkpoint_dir}")
            return checkpoint_dir, is_completed  # checkpoint found!
        print("isdir is false")
        return None, False  # first training

    def query(self, model_name: str, input_text: str, already_loaded_model=None, adapter_path: str = None,
              bit: int = None) -> tuple[str, PeftModel]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        max_new_tokens = 300
        temperature = 1e-9

        if bit:
            print(f"    Model '{model_name}' is quantized with {CYAN}{bit} bits.{RESET}")

        if already_loaded_model is None:
            already_loaded_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True if bit == 4 else False,
                    load_in_8bit=True if bit == 8 else False,
                    # bnb_4bit_compute_dtype=torch.bfloat16,
                    # bnb_4bit_use_double_quant=True,
                    # bnb_4bit_quant_type='nf4',
                ) if bit else None,
                # low_cpu_mem_usage=True
            )
            if adapter_path:
                adapter_path, _ = CodeLlamaInstructCoder.get_last_checkpoint(adapter_path)
                print(f"    Loading {CYAN}adapter{RESET} from '{adapter_path}'")
                already_loaded_model = PeftModel.from_pretrained(already_loaded_model, adapter_path, offload_folder="finetuning/lora/offload/codellama_python")
        already_loaded_model.config.use_cache = False
        already_loaded_model.eval()

        if bool(re.search(r"CodeLlama-\d+b-Instruct-hf", model_name)):
            print(f"Formatting prompt for {CYAN}Instruct{RESET} model.")
            prompt = f"<s>[INST] {input_text} [/INST]"
        elif bool(re.search(r"CodeLlama-\d+b-Python-hf", model_name)):
            print(f"Formatting prompt for {CYAN}Python{RESET} model.")
            prompt = "<s>" + input_text

            # print(f"Input:____{prompt}____")

            input_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
            with torch.cuda.amp.autocast():
                generation_output = already_loaded_model.generate(
                    input_ids=input_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=10,
                    top_p=0.9,
                    temperature=1e-9,
                    # repetition_penalty=1.1, # needs to be > 1?
                    # num_return_sequences=1, # no effect
                    # eos_token_id=tokenizer.eos_token_id,
                )

            op = tokenizer.decode(generation_output[0], skip_special_tokens=True) # if True, doesn't print <s> and </s>
            print(f"Text:>>>>{op}<<<<")

            # prompt = prompt[3:]
            # lines_beginning = prompt.split('\n')
            # lines_full = op.split('\n')
            #
            # for i, line in enumerate(lines_full):
            #     if i > 3 and i >= len(lines_beginning) or lines_full[i] != lines_beginning[i]:
            #         lines_full.pop(i)
            #         break
            #
            # op = '\n'.join(lines_full)

            return op, already_loaded_model

        elif bool(re.search(r"CodeLlama-\d+b-hf", model_name)):
            print(f"Formatting prompt for {CYAN}Base{RESET} model.")
            prompt = "<s>" + input_text
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized [coder_llms.py].")

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
            print("Text:****\n", text, "****")

            if bool(re.search(r"CodeLlama-\d+b-Instruct-hf", model_name)):
                text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
            return text

        return generate(already_loaded_model, prompt), already_loaded_model


class CodeLlamaBaseCoder(CoderLLM):

    def query(self, model_name: str, input_text: str, already_loaded_model=None, adapter_path: str = "",
              bit: int = None) -> str:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if bit:
            print(f"    Model '{model_name}' is quantized with {CYAN}{bit} bits.{RESET}")

        if already_loaded_model is None:
            already_loaded_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True if bit == 4 else False,
                    load_in_8bit=True if bit == 8 else False,
                    # bnb_8bit_compute_dtype=torch.float16 if bit == "8" else torch.bfloat16,
                ),
            )
            if adapter_path != "":
                adapter_path, _ = CodeLlamaInstructCoder.get_last_checkpoint(adapter_path)
                already_loaded_model = PeftModel.from_pretrained(already_loaded_model, adapter_path)

        already_loaded_model.eval()

        max_new_tokens = 500
        temperature = 1e-9

        prompt = input_text

        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        output = already_loaded_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        parts = prompt.split("<FILL_ME>")
        removed_fill_me = parts[0] + parts[1]
        first_n_chars = len(removed_fill_me)
        text = text[first_n_chars:]
        final_result_function = parts[0] + text + parts[1]

        return final_result_function, already_loaded_model


class MagiCoder(CoderLLM):
    def query(self, model_name: str, input_text: str, already_loaded_model=None, adapter_path: str = None,
              bit: int = None) -> tuple[str, PeftModel]:

        if not already_loaded_model:
            generator = pipeline(
                model=model_name,
                task="text-generation",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            generator = already_loaded_model

        result = generator(input_text, max_length=1024, num_return_sequences=1, temperature=0.0)
        print(">>>" + result[0]["generated_text"] + "<<<")
        return result[0]["generated_text"], generator


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
            stop_tokens = ['<|endoftext|>']  # for 1B, 3B or 15B models
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


class Magicoder(CoderLLM):

    def query(self, model_name: str, input_text: str) -> str:
        prompt = f'''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

        @@ Instruction
        {input_text}

        @@ Response
        '''

        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2048)

        llm = LLM(model="TheBloke/Magicoder-S-DS-6.7B-AWQ", quantization="AWQ", dtype="auto",
                  gpu_memory_utilization=0.95, max_model_len=8192)

        outputs = llm.generate(prompt, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            print(type(generated_text))
            return generated_text
