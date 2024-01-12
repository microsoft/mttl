from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskModule,
    FlatMultiTaskConfig,
    HeldOutFlatMultiTaskModule,
)
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
import json
import argparse
import os
from collections import defaultdict
import random
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    type=str,
    default="",
    help="input file",
)

parser.add_argument(
    "--output_file",
    type=str,
    default="module_pairwise_dataset.json",
    help="output file",
)

parser.add_argument(
    "--transfer_matrix_path",
    type=str,
    default="",
    help="transfer_matrix_path",
)

# if create pair dataset or pair dataset with input
parser.add_argument(
    "--create_pair_dataset",
    type=bool,
    default=False,
    help="create_pair_dataset",
)


def get_pair_dataset(args):
    """
    Generate a pairwise dataset based on the given arguments. For each each task,
    we want to creat a triple pair (task_eval_on, m_positive, m_negative).

    Args:
        args (list): A list of arguments specifying the dataset generation parameters.
        The file input is a jsonl file with the following format:
        {"expert_name": "default", "task_eval_on": "ARB", "score": 5.155895955860615}

    Returns:
        dataset (list): A list of pairs representing the generated dataset.
        (task_eval_on, m_positive, m_negative)
    """

    data = []
    files = os.listdir(args.transfer_matrix_path)
    for file in files:
        if file.endswith(".jsonl"):
            file_path = os.path.join(args.transfer_matrix_path, file)
            with open(file_path, "r") as file:
                for line in file:
                    data.append(json.loads(line))

    # Step 1: Identify the Positive Expert for Each Task
    positive_experts = defaultdict(lambda: {"expert_name": None, "score": float("inf")})

    for entry in data:
        task = entry["task_eval_on"]
        score = entry["score"]
        expert = entry["expert_name"]

        if score < positive_experts[task]["score"]:
            positive_experts[task] = {"expert_name": expert, "score": score}

    # Step 2: Create a Pool of Positive Experts
    positive_expert_pool = set(
        [expert["expert_name"] for expert in positive_experts.values()]
    )

    # Step 3: Select Negative Experts for Each Task
    negative_experts_for_tasks = {}

    for task in positive_experts:
        # Ensure the positive expert is not selected as a negative expert
        available_experts = list(
            positive_expert_pool - {positive_experts[task]["expert_name"]}
        )
        # Randomly select 5 negative experts
        negative_experts_for_tasks[task] = random.sample(
            available_experts, min(5, len(available_experts))
        )

    # Prepare the final dataset to be displayed
    final_dataset = []
    for task, info in positive_experts.items():
        final_dataset.append(
            {
                "task_eval_on": task,
                "positive_expert_name": info["expert_name"],
                "negative_expert_name": negative_experts_for_tasks[task],
            }
        )
    print(final_dataset[:5])  # Displaying first 5 entries of the final dataset

    dataset_for_export = []

    for entry in final_dataset:
        for neg_expert in entry["negative_expert_name"]:
            dataset_for_export.append(
                {
                    "task_eval_on": entry["task_eval_on"],
                    "positive_expert_name": entry["positive_expert_name"],
                    "negative_expert_name": neg_expert,
                }
            )

    df = pd.DataFrame(dataset_for_export)
    output_file_path = args.output_file
    df.to_csv(output_file_path, index=False)


def get_pairwise_dataset_with_input(args):
    import pandas as pd
    import json
    import tqdm

    #   for each input x, we want to creat a triple pair (x, m_positive, m_negative)

    # filter the dataset witl only the finetune_task_name
    if args.input_file.endswith(".jsonl"):
        df = pd.read_json(args.input_file, lines=True)
    elif args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
    fout = open(args.output_file, "w")

    for task in tqdm.tqdm(df["task_eval_on"].unique()):
        df_task = df[df["task_eval_on"] == task]
        # df_task_top = df_task.groupby(["eval_task", "m1"]).head(1)
        # print(df_task_top)

        data_module = HeldOutFlatMultiTaskModule(
            FlatMultiTaskConfig(
                dataset="sordonia/adauni-v1-flat",
                model="EleutherAI/gpt-neo-125m",
                finetune_task_name=task,
                predict_batch_size=10,
            ),
            for_generation=True,
        )
        # data_module = MMLUDataModule(
        #     MMLUDataConfig(
        #         "mmlu",
        #         model="t5-small",
        #         model_family="seq2seq",
        #         train_batch_size=4,
        #         predict_batch_size=4,
        #         finetune_task_name=task,
        #     )
        # )
        test_dataset = data_module.test_dataloader()
        print("task::{}....len:{}".format(task, len(test_dataset)))

        for batch in test_dataset:
            for input_text in batch["sources_texts"]:
                for e, element in df_task.iterrows():
                    fout.write(
                        json.dumps(
                            {
                                "eval_task": element["task_eval_on"],
                                "sources_texts": input_text,
                                "positive_expert_names": element[
                                    "positive_expert_name"
                                ],
                                "negative_expert_names": element[
                                    "negative_expert_name"
                                ],
                            }
                        )
                        + "\n"
                    )
            fout.flush()
            ## if only need subset of the dataset
            # break


if __name__ == "__main__":
    args = parser.parse_args()
    if args.create_pair_dataset:
        get_pair_dataset(args)
    else:
        get_pairwise_dataset_with_input(args)
