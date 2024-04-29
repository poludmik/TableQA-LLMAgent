from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import os
from utils import *

seed = 1337

model_name = "codellama/CodeLlama-7b-Python-hf"
# model_name = "facebook/opt-350m"
# quantization_config = BitsAndBytesConfig(load_in_4bit=False)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            #  quantization_config=quantization_config,
                                             device_map="auto"
                                             )

model = activate_adapter(model, "/home/poludmik/2024/AgentToBeNamed/finetuning/lora/cp/pretrained_on_pandas/checkpoint-2000")


model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)


# peft_model_id = "finetuning/lora/cp/another_codellama_4targets/checkpoint-500"
# peft_model_id = "/home/micha/homeworks/NLP/fine-tuning/qlora_for_codegen/completion_finetuning_data/output/checkpoint-250"
peft_model_id = "finetuning/lora/cp/upd_codellama_gate_only_lr2"

peft_model = PeftModel.from_pretrained(model, peft_model_id, offload_folder="finetuning/lora/offload/codellama_python")
# peft_model = model
peft_model.eval()



repo = "poludmik/code_completion_for_data_analysis"
dataset = load_dataset(repo, split="train_upd")  # everything is in the train split on HF
split_ratio = 0.05  # 5% for validation
num_validation_samples = int(len(dataset) * split_ratio)
split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)
small_dataset = split_dataset['test'].select(range(20))
train_dataset = split_dataset['train']
validation_dataset = small_dataset

# input_prompt = "print('first text')\n"

input_prompt = train_dataset[500]["input"] # + "# Code:\n    "
input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
# print(input_tokens)
with torch.cuda.amp.autocast():
    generation_output = peft_model.generate(
        input_ids=input_tokens,
        max_new_tokens=300,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=1e-9,
        # repetition_penalty=1.1, # needs to be > 1?
        # num_return_sequences=1, # no effect
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=0,
      )

op = tokenizer.decode(generation_output[0], skip_special_tokens=False)
print(">>>>"+op+"<<<<")

