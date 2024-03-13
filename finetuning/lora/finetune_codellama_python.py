import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
from trl import SFTTrainer

seed = 228
eos_token = "</s>"

model_name = "codellama/CodeLlama-7b-Python-hf"

quantization_config = BitsAndBytesConfig(load_in_4bit=False)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=quantization_config,
                                             device_map="auto"
                                             )

tokenizer = AutoTokenizer.from_pretrained(model_name)

repo = "poludmik/code_completion_for_data_analysis"
dataset = load_dataset(repo, split="train")  # everything is in the train split on HF
# print(dataset["train"][0])

split_ratio = 0.1  # 10% for validation
num_validation_samples = int(len(dataset) * split_ratio)

split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)

# Access the training and validation sets
train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(validation_dataset))


# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['input'])):
#         text = f"{example['input'][i]}{example['output'][i]}{eos_token}"
#         output_texts.append(text)
#     return output_texts

def generate_prompt(dialogue, summary=None):
    input = f"{dialogue}\n"
    summary = f"{summary + eos_token if summary else ''} "
    prompt = input + summary
    return prompt


# print(generate_prompt(train_dataset[0]["input"], train_dataset[0]["output"]))

# input_prompt = generate_prompt(train_dataset[50]["input"])
# tokenized = tokenizer(input_prompt, return_tensors="pt")
# input_tokens = tokenized["input_ids"]
# attention_mask = tokenized["attention_mask"]
# with torch.cuda.amp.autocast():
#     generation_output = model.generate(
#         input_ids=input_tokens,
#         attention_mask=attention_mask,
#         max_new_tokens=100,
#         do_sample=True,
#         top_k=10,
#         top_p=0.9,
#         temperature=0.01,
#         repetition_penalty=1.15,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id = tokenizer.eos_token_id
#         )
#
# op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
# print(op)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


def print_trainable_parameters(m):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in m.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}%"
    )


print_trainable_parameters(model)

output_dir = "finetuning/lora/cp/"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = 10
warmup_ratio = 0.03
evaluation_strategy = "steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    evaluation_strategy=evaluation_strategy,
    save_steps=save_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    eval_accumulation_steps=eval_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
)


def formatting_func(prompt):
    output = []
    for d, s in zip(prompt["input"], prompt["output"]):
        op = generate_prompt(d, s)
        output.append(op)
    return output


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=250,
    tokenizer=tokenizer,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/final")
