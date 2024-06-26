"""
This script doesn't work properly, use another_codellama_finetune.py instead. 
This script is kept for reference purposes only.
"""

import torch
import transformers
import argparse

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import wandb
from utils import *
# import hydra
import yaml
import time

def main(config_path: str = "config.yaml"):
    cfg = DotDict(load_config(config_path))
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               config=dict(cfg),
               job_type="training",
               name=cfg.hyperparameters.run_name + ": " + get_readable_datetime_string(),
               )

    seed = cfg.hyperparameters.seed

    model_name = "codellama/CodeLlama-7b-Python-hf"
    # model_name = "facebook/opt-350m"

    quantization_config = BitsAndBytesConfig(load_in_8bit=False)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                #  quantization_config=quantization_config,
                                                 device_map="auto"
                                                 )


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"{YELLOW}Model and tokenizer loaded{RESET}")

    repo = "poludmik/code_completion_for_data_analysis"
    # dataset = load_dataset("imdb", split="train")
    dataset = load_dataset(repo, split="train")  # everything is in the train split on HF
    # print(dataset["train"][0])

    split_ratio = cfg.hyperparameters.split_ratio  # 1% for validation
    num_validation_samples = int(len(dataset) * split_ratio)

    split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)
    # small_dataset = split_dataset['test'].select(range(40))

    # Access the training and validation sets
    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']
    # validation_dataset = small_dataset
    print(validation_dataset)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(validation_dataset))

    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['input'])):
    #         text = f"{example['input'][i]}{example['output'][i]}{eos_token}"
    #         output_texts.append(text)
    #     return output_texts


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
        r=cfg.hyperparameters.lora_config.r,
        lora_alpha=cfg.hyperparameters.lora_config.alpha,
        lora_dropout=cfg.hyperparameters.lora_config.dropout,
        target_modules=cfg.hyperparameters.lora_config.target_modules,
        bias=cfg.hyperparameters.lora_config.bias,
        task_type=cfg.hyperparameters.lora_config.task_type,
    )

    # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.pad_token = tokenizer.eos_token
    # model.resize_token_embeddings(len(tokenizer))

    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {YELLOW}{trainable}{RESET} | total: {BLUE}{total}{RESET} | Percentage: {YELLOW}{trainable / total * 100:.4f}{RESET}%")

    training_args = transformers.TrainingArguments(
        output_dir=cfg.hyperparameters.output_dir,
        per_device_train_batch_size=cfg.hyperparameters.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.hyperparameters.gradient_accumulation_steps,
        optim=cfg.hyperparameters.optim,
        evaluation_strategy=cfg.hyperparameters.evaluation_strategy,
        save_steps=cfg.hyperparameters.save_steps,
        learning_rate=cfg.hyperparameters.learning_rate,
        logging_steps=cfg.hyperparameters.logging_steps, # how often to log to W&B
        max_grad_norm=cfg.hyperparameters.max_grad_norm,
        max_steps=cfg.hyperparameters.max_steps,
        warmup_ratio=cfg.hyperparameters.warmup_ratio,
        group_by_length=cfg.hyperparameters.group_by_length,
        lr_scheduler_type=cfg.hyperparameters.lr_scheduler_type,
        ddp_find_unused_parameters=cfg.hyperparameters.ddp_find_unused_parameters,
        eval_accumulation_steps=cfg.hyperparameters.eval_accumulation_steps,
        per_device_eval_batch_size=cfg.hyperparameters.per_device_eval_batch_size,
        report_to="wandb",  # enable logging to W&B
        # run_name=cfg.hyperparameters.run_name + "_id" + str(random.randint(0, 1000000)),  # name of the W&B run (optional)
        run_name=cfg.hyperparameters.run_name,
    )

    response_template = """# Code:"""
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=cfg.hyperparameters.trainer.max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args
    )


    # Instantiate the WandbPredictionProgressCallback
    # progress_callback = WandbPredictionProgressCallback(
    #     trainer=trainer,
    #     tokenizer=tokenizer,
    #     val_dataset=validation_dataset,
    #     # num_samples=cfg.hyperparameters.callback_nth, # always select nth sample from the validation dataset
    # )

    # Add the callback to the trainer
    # trainer.add_callback(progress_callback)

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    print(f"{YELLOW}Starting training{RESET}")
    trainer.train()
    trainer.save_model(f"{cfg.hyperparameters.output_dir}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="finetuning/lora/config/LoRA_params.yaml", help="Path to the config file.")
    main(parser.parse_args().config)
