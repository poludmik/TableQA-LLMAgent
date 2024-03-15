from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch

seed = 1337

model_name = "codellama/CodeLlama-7b-Python-hf"
# model_name = "facebook/opt-350m"

quantization_config = BitsAndBytesConfig(load_in_4bit=False)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=quantization_config,
                                             device_map="auto"
                                             )

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_model_id = "finetuning/lora/cp/codellama_python/final"
peft_model = PeftModel.from_pretrained(model, peft_model_id, offload_folder="finetuning/lora/offload/test_opt")
# peft_model = model

repo = "poludmik/code_completion_for_data_analysis"
dataset = load_dataset(repo, split="train")  # everything is in the train split on HF
split_ratio = 0.5  # 10% for validation
num_validation_samples = int(len(dataset) * split_ratio)
split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)
small_dataset = split_dataset['test'].select(range(20))
train_dataset = split_dataset['train']
validation_dataset = small_dataset


input_prompt = train_dataset[2]["input"] + "# Code:\n    "
# input_prompt = """\"I Am Curious: Yellow\" is a risible and pretentious steaming"""
input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
with torch.cuda.amp.autocast():
    generation_output = peft_model.generate(
        input_ids=input_tokens,
        max_new_tokens=100,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.3,
        repetition_penalty=1.15,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
      )

op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(">>>>"+op+"<<<<")

