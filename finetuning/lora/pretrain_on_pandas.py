import argparse
import random
import wandb
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, IntervalStrategy 

from utils import *


def main(config_path: str = "config.yaml"):
    cfg = DotDict(load_config(config_path))

    seed = cfg.hyperparameters.seed
    random.seed(seed)
    model_name = "codellama/CodeLlama-7b-Python-hf"

    model = LlamaForCausalLM.from_pretrained(model_name)
    model.use_cache = False
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)   
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            # "pad_token": "[PAD]",
    })
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0 # <unk>, llama doesn't have a padding token?
    tokenizer.padding_side = "left"

    # datasets = load_dataset(cfg.hf.repo, split="pandas_documentation_examples")  # everything is in the train split on HF

    dataset1 = load_dataset(cfg.hf.repo, split="pandas_documentation_examples")
    dataset2 = load_dataset(cfg.hf.repo, split="DS1000")
    dataset3 = load_dataset(cfg.hf.repo, split="OSSInstruct")
    dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3])
    dataset = dataset.remove_columns("text")
    dataset = dataset.shuffle(seed=seed)

    def tokenize_function(examples):
        texts = list(map(lambda x: x, examples["code"]))
        # toks = tokenizer(texts, truncation=True, max_length=cfg.hyperparameters.trainer.max_seq_length, padding='max_length')
        toks = tokenizer(texts)
        return toks

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda x: {'labels':x['input_ids']})

    num_validation_samples = int(len(dataset) * cfg.hyperparameters.split_ratio)
    split_dataset = dataset.train_test_split(test_size=num_validation_samples, seed=seed)

    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    lora_config = LoraConfig(
        r=cfg.hyperparameters.lora_config.r,
        lora_alpha=cfg.hyperparameters.lora_config.alpha,
        target_modules=cfg.hyperparameters.lora_config.target_modules,
        lora_dropout=cfg.hyperparameters.lora_config.dropout,
        bias=cfg.hyperparameters.lora_config.bias,
        task_type=cfg.hyperparameters.lora_config.task_type,
    )

    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               config=dict(cfg),
               job_type="training",
               name=cfg.hyperparameters.run_name + ": " + get_readable_datetime_string(),
               )


    training_args = TrainingArguments(
        output_dir=cfg.hyperparameters.output_dir,  # The output directory
        overwrite_output_dir=cfg.hyperparameters.overwrite_output_dir,  # overwrite the content of the output directory
        evaluation_strategy=cfg.hyperparameters.evaluation_strategy,
        max_steps=cfg.hyperparameters.max_steps, # instead of epochs
        per_device_train_batch_size=cfg.hyperparameters.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=cfg.hyperparameters.per_device_eval_batch_size,  # batch size for evaluation
        gradient_accumulation_steps=cfg.hyperparameters.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.hyperparameters.eval_accumulation_steps,
        save_steps=cfg.hyperparameters.save_steps,
        learning_rate=cfg.hyperparameters.learning_rate,
        logging_steps=cfg.hyperparameters.logging_steps, # how often to log to W&B
        eval_steps=cfg.hyperparameters.eval_steps,  # Number of update steps between two evaluations.
        warmup_ratio=cfg.hyperparameters.warmup_ratio,  # number of warmup steps for learning rate scheduler
        # lr_scheduler_kwargs={"min_lr_ratio": cfg.hyperparameters.min_lr_ratio},
        weight_decay = cfg.hyperparameters.weight_decay,
        fp16=cfg.hyperparameters.fp16,
        lr_scheduler_type = cfg.hyperparameters.lr_scheduler_type,
        report_to="wandb",
        gradient_checkpointing=cfg.hyperparameters.gradient_checkpointing,
        )

    trainer = Trainer(model=model, args=training_args, 
                      train_dataset=train_dataset, 
                      eval_dataset=validation_dataset,
                      )

    trainer.train()
    model.save_pretrained(cfg.hyperparameters.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="finetuning/lora/config/another_LoRA_params.yaml", help="Path to the config file.")
    main(parser.parse_args().config)
