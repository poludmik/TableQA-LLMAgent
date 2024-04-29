"""
This script is used to merge the adapter and push the model to the Hugging Face model hub.
"""
import argparse
import random
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, IntervalStrategy 
from utils import *
from huggingface_hub import Repository


model_name = "codellama/CodeLlama-7b-Python-hf"

model = LlamaForCausalLM.from_pretrained(model_name)
peft_model_id = "/home/poludmik/2024/AgentToBeNamed/finetuning/lora/cp/pretrained_on_pandas/checkpoint-2000"
model = PeftModel.from_pretrained(model, peft_model_id)
model.merge_adapter()

# model.merge_and_unload()
print(model)

# save the model
# model.save_model("finetuning/lora/merged_models/CL7BPy_Pandas2000_Apr27")

# push the model to the Hugging Face model hub
# model.push_to_hub(commit_message="CL7B_Python on 2000 pandas examples LoRA adapter")
repo = Repository(local_dir="finetuning/lora/merged_models/CL7BPy_Pandas2000_Apr27", clone_from="poludmik/CL7B_Python_2000pandas_LoRA")

# Commit and push all files in `local_repo_path` to Hugging Face under your specific repository
repo.push_to_hub(commit_message="Upload LoRA fine-tuned model")

