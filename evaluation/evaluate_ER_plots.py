from agenttobenamed import AgentTBN
import time
import os
import pandas as pd
import random
import hydra

"""
  tables_dataset_dir: dataset/dataset_tables/
  excel: dataset/dataset_82_gpt35_functions.xlsx
  max_debug_times: 0
  head_number: 2
  add_column_description: False
  n_column_samples: 2

  tagging_strategy: openai
  query_type: plot

  propmt_strategy: coder_only_functions

  gpt_model: gpt-3.5-turbo-1106

  coder_model: gpt-3.5-turbo-1106
  coder_adapter_path: ""
  coder_quantization_bits:

  debug_model: gpt-3.5-turbo-1106
  debug_adapter_path: ""
  debug_quantization_bits:
  debug_strategy: basic

  output_dir: evaluation/plots_ER/
  output_filename: plots_ER_gpt35_0debug_colsampl2.xlsx

  skip_reasoning: False
"""


@hydra.main(config_path="conf", config_name="config_ER_plots_gpt35turbo_save.yaml", version_base="1.1")
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

    agent = AgentTBN("",  # csv_path,
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

    executable_250 = []
    executable_t2a = []

    for index, row in dataset_df.iterrows():
        try:
            if row["user_query"] == "" or row["user_query"] == "nan":  # for the last row with plot counts
                continue

            agent.load_new_df(os.path.join(tables_dataset_dir, row["table_name"]))

            if skip_reasoning:
                agent.skip_reasoning_part(row["plan"], row["tagged_query_type"], row["prompt_user_for_planner"])
                save_plot_to = row["plot_filename"]
            else:
                save_plot_to = plots_dir + os.path.splitext(row["table_name"])[0] + str(index) + "-" + str(random.randint(0, 9)) + ".png"

            result, details = agent.answer_query(row["user_query"], save_plot_path=save_plot_to, show_plot=show_plot)

            if row["text2analysis"] == "True":
                executable_t2a.append(1 if details["successful_code_execution"] == "True" else 0)
            else:
                executable_250.append(1 if details["successful_code_execution"] == "True" else 0)

            # If save_plot_to file exists, copy it to the plots_dir_save/ and remove from plots_dir/
            if os.path.exists(save_plot_to):
                os.system(f"cp {save_plot_to} {plots_dir_save}")
                os.system(f"rm {save_plot_to}")

            for key in details.keys():
                dataset_df.loc[index, key] = details[key]

            dataset_df.to_excel(save_dataset_path, index=False)

        except Exception as e:
            print("Exception:", e)
            print("Exception at index:", index, "row:", row)
            continue

    dataset_df.to_excel(save_dataset_path, index=False)

    # create a txt file with the results:
    filename = os.path.join(output_dir, "ERs.txt")
    with open(filename, "w") as f:
        if len(executable_250) == 0:
            f.write("No queries from 250\n")
            executable_250 = [0]
        else:
            f.write(f"Number of queries from 250: {len(executable_250)}\n")
            f.write(f"Number of executable queries from 250: {sum(executable_250)}\n")

        if len(executable_t2a) == 0:
            f.write("No queries from text2analysis\n")
            executable_t2a = [0]
        else:
            f.write(f"Number of queries from text2analysis: {len(executable_t2a)}\n")
            f.write(f"Number of executable queries from text2analysis: {sum(executable_t2a)}\n")

        f.write(f"----------\n")
        # percentage of executable queries
        f.write(f"Percentage of queries from 250 that are executable: {sum(executable_250) / len(executable_250)}\n")
        f.write(f"Percentage of queries from text2analysis that are executable: {sum(executable_t2a) / len(executable_t2a)}\n")

    print(f"Elapsed time: {time.time() - start_time} seconds")



if __name__ == "__main__":
    main()
