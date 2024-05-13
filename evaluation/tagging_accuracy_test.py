import sys
import os

sys.path.append(os.path.abspath('.'))

from tableqallmagent import LLMAgent
import time
import pandas as pd
import random
import hydra
import argparse
from tableqallmagent.logger import *

"""
Run this with `python3 evaluation/tagginh_accuracy_test.py --config-name config_kek.yaml`
"""

@hydra.main(config_path="conf", config_name="Provide a config.yaml!", version_base="1.1")
def main(cfg):
    tables_dataset_dir = cfg.params.tables_dataset_dir
    clean_excel = cfg.params.excel_with_queries

    max_debug_times = cfg.params.max_debug_times
    head_number = cfg.params.head_number
    add_column_description = cfg.params.add_column_description
    n_column_samples = cfg.params.n_column_samples

    tagging_strategy = cfg.params.tagging_strategy
    query_type = cfg.params.query_type

    propmt_strategy = cfg.params.propmt_strategy

    gpt_model = cfg.params.gpt_model

    coder_model = cfg.params.coder_model
    coder_adapter_path = cfg.params.coder_adapter_path
    coder_quantization_bits = cfg.params.coder_quantization_bits

    debug_model = cfg.params.debug_model
    debug_adapter_path = cfg.params.debug_adapter_path
    debug_quantization_bits = cfg.params.debug_quantization_bits
    debug_strategy = cfg.params.debug_strategy

    output_dir = cfg.params.output_dir
    output_filename = cfg.params.output_filename
    skip_reasoning = cfg.params.skip_reasoning

    show_plot = cfg.params.show_plot

    plots_dir = "plots/" # will be in the outputs/ folder (hydra logs), serves as a tmp buffer for plots

    original_working_dir = os.path.dirname(os.path.dirname(__file__))  # because hydra changes the working directory
    output_dir = os.path.join(original_working_dir, output_dir)
    tables_dataset_dir = os.path.join(original_working_dir, tables_dataset_dir)
    clean_excel = os.path.join(original_working_dir, clean_excel)
    plots_dir_save = os.path.join(output_dir, plots_dir)

    print("original_working_dir:", original_working_dir)
    print("output_dir:", output_dir)
    print("tables_dataset_dir:", tables_dataset_dir)
    print("clean_excel:", clean_excel)
    print("plots_dir_save:", plots_dir_save)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir_save):
        os.makedirs(plots_dir_save)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    start_time = time.time()

    save_dataset_path = os.path.join(output_dir, output_filename)

    dataset_df = pd.read_excel(clean_excel).astype(str)

    agent = LLMAgent("",  # csv_path,
                     max_debug_times=max_debug_times,
                     head_number=head_number,
                     add_column_description=add_column_description,
                     n_column_samples=n_column_samples,
                     tagging_strategy=tagging_strategy,
                     query_type=query_type,
                     prompt_strategy=propmt_strategy,
                     coder_model=coder_model,
                     coder_adapter_path=coder_adapter_path,
                     coder_quantization_bits=coder_quantization_bits,
                     debug_model=debug_model,
                     debug_adapter_path=debug_adapter_path,
                     debug_quantization_bits=debug_quantization_bits,
                     debug_strategy=debug_strategy,
                     gpt_model=gpt_model,
                     )

    results = []
    tags_on_plots = []
    tags_on_general = []

    for index, row in dataset_df.iterrows():
        if row["user_query"] == "nan" or not row["user_query"]:
            print("Skipping empty query at index", index)
            continue

        # tag a query and compare it with the ground truth
        query = row["user_query"]
        ground_truth = True if row["has_plot_answer"] == "1" else False

        tagged_query = agent.tag(query)
        print(f"ground_truth: {ground_truth}")
        print(f"tagged_query: {tagged_query == 'plot'}")
        results.append(int((tagged_query == "plot") == ground_truth))

        if ground_truth:
            tags_on_plots.append(int(tagged_query == "plot"))
        else:
            tags_on_general.append(int(tagged_query == "general"))

        print(f"{GREEN}True :){RESET}\n" if results[-1] else f"{RED}False >:({RESET}\n")

    print(f"Elapsed time: {time.time() - start_time} seconds")
    print(f"Number of queries: {len(results)}")
    print(f"Accuracy: {YELLOW}{sum(results) / len(results)}{RESET}")
    print(f"Accuracy on plots: {MAGENTA}{sum(tags_on_plots) / len(tags_on_plots)}{RESET}")
    print(f"Accuracy on general: {MAGENTA}{sum(tags_on_general) / len(tags_on_general)}{RESET}")


if __name__ == "__main__":
    main()
