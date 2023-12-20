from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskModule,
    FlatMultiTaskConfig,
)
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig

import argparse

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


def get_pairwise_dataset(args):
    import pandas as pd
    import json
    import tqdm

    #   for each input x, we want to creat a triple pair (x, m_positive, m_negative)

    # filter the dataset witl only the finetune_task_name
    df = pd.read_csv(args.input_file)
    fout = open(args.output_file, "w")

    for task in df["task_eval_on"].unique():
        df_task = df[df["task_eval_on"] == task]
        # df_task_top = df_task.groupby(["eval_task", "m1"]).head(1)
        # print(df_task_top)

        data_module = FlatMultiTaskModule(
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

        for batch in tqdm.tqdm(test_dataset):
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
            break


if __name__ == "__main__":
    args = parser.parse_args()
    get_pairwise_dataset(args)
