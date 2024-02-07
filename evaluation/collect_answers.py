from agenttobenamed import AgentTBN
import time
import os
import pandas as pd
import random
import hydra


@hydra.main(config_path="conf", config_name="config_collect_private.yaml", version_base="1.1")
def main(cfg):
    output_dir = cfg.params.output_dir
    tables_dataset_dir = cfg.params.tables_dataset_dir
    clean_excel = cfg.params.excel
    max_debug_times = cfg.params.max_debug_times
    reasoning_llm = cfg.params.reasoning_llm
    coding_llm = cfg.params.coding_llm
    head_number = cfg.params.head_number
    output_filename = cfg.params.output_filename
    adapter_path = cfg.params.adapter_path

    plots_dir = "plots/" # will be in the outputs/ folder (hydra logs), serves as a tmp buffer for plots

    original_working_dir = os.path.dirname(os.path.dirname(__file__))  # because hydra changes the working directory
    output_dir = os.path.join(original_working_dir, output_dir)
    tables_dataset_dir = os.path.join(original_working_dir, tables_dataset_dir)
    clean_excel = os.path.join(original_working_dir, clean_excel)
    plots_dir_save = os.path.join(output_dir, plots_dir)
    # print("plots_dir_save:", plots_dir_save)
    # print("original_working_dir:", original_working_dir)
    # print("output_dir:", output_dir)
    # print("tables_dataset_dir:", tables_dataset_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir_save):
        os.makedirs(plots_dir_save)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    start_time = time.time()

    # llm_model = "gpt-4-1106-preview"  # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
    # save_dataset_path = f"dataset_changed_{llm_model}_hn-{head_number}_mdt-{max_debug_times}_7".replace('.', '_') + ".xlsx"
    save_dataset_path = os.path.join(output_dir, output_filename)

    dataset_df = pd.read_excel(clean_excel).astype(str)
    # print(dataset_df)

    # q_idx = list(range(13, 14))
    # q_idx = [0, 1, 2, 3, 4]
    prev_table_name = None
    for index, row in dataset_df.iterrows():
        try:
            # if index in q_idx:
            #     continue
            if row["user_query"] == "" or row["user_query"] == "nan":  # for the last row with plot counts
                continue

            if row["table_name"] != prev_table_name:
                prev_table_name = row["table_name"]
                print("Processing table:", prev_table_name)
                agent = AgentTBN(tables_dataset_dir + row["table_name"],
                                 max_debug_times=max_debug_times,
                                 gpt_model=reasoning_llm,
                                 coder_model=coding_llm,
                                 head_number=head_number,
                                 adapter_path=adapter_path
                                 )


            save_plot_to = plots_dir + os.path.splitext(row["table_name"])[0] + str(index) + "-" + str(random.randint(0, 9)) + ".png"

            result, details = agent.answer_query(row["user_query"], save_plot_path=save_plot_to)

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
    print(f"Elapsed time: {time.time() - start_time} seconds")



if __name__ == "__main__":
    main()
