import argparse
import os 
import torch
import numpy as np
import pandas as pd
import time
import json 
from tqdm import tqdm
import time
import sys
import click

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from projects.instr_routing.eval.model_dict import model_dict

import glob
from projects.instr_routing.eval.mmlu.categories import subcategories, categories
from projects.instr_routing.eval.utils import (
    get_next_word_predictions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
)
from projects.instr_routing.eval.ni.gen_ni_predictions import (
    load_model_for_generation,
    dict_to_dataclass,
)

# from eval.model_loader import (
#     load_from_llama,
#     load_from_mttl,
#     load_from_peft,
# )


choices = ["A", "B", "C", "D"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model(
    args,
    subject,
    model,
    tokenizer,
    dev_df,
    test_df,
    batch_size=1,
    topic_router=None,
    skill_selector=None,
    cluster_depth=1,
):
    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        # make sure every prompt is less than 2048 tokens
        while tokenized_prompt.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            tokenized_prompt = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids

        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt.strip() + "\n<|assistant|>\nThe answer is:"

        prompts.append(prompt)

    # get the answer for all examples
    # note: here we cannot directly use convert_tokens_to_ids because the some tokenizers will automatically add space prefix.
    answer_choice_ids = [
        tokenizer.encode(answer_choice, add_special_tokens=False)[0]
        for answer_choice in choices
    ]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=batch_size,
        topic_router=topic_router,
        skill_selector=skill_selector,
        cluster_depth=cluster_depth,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    import tiktoken

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [
        gpt_tokenizer.encode(" " + x)[0] for x in choices
    ]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(
        [[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]
    )  # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


@click.command()
@click.option("--ntrain", type=int, default=5)
@click.option("--data_dir", type=str, default="/home/v-oostapenko/data/mmlu")
@click.option("--save_dir", type=str, default="results/mmlu/llama-7B/")
@click.option(
    "--model_name",
    type=str,
    default="alpaca_smear_12_xr4_cos",
    help="if specified, we will load the model to generate the predictions.",
)
@click.option(
    "--model_path",
    type=str,
    default="/home/v-oostapenko/dev/amlt//alpaca_smear/alpaca_smear_12_xr4_cos/yahma_llama-7b-hfnekcmtyy_alpaca_lora_cbtm_dense-val/loss=0.8796.ckpt",
    help="if specified, we will load the model to generate the predictions.",
)
@click.option(
    "--openai_engine",
    type=str,
    default=None,
    help="if specified, we will use the OpenAI API to generate the predictions.",
)
@click.option(
    "--n_instances",
    type=int,
    default=None,
    help="if specified, a maximum of n_instances per subject will be used for the evaluation.",
)
@click.option(
    "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
)
@click.option(
    "--skill_selector",
    type=str,
    default="poly",
    help="skill selector",
)
@click.option("--amlt_experiment_name", type=str, default="alpaca_smear")
@click.option("--use_chat_format", is_flag=False)
@click.option("--subjects", type=str, default="-1")
def main(
    ntrain=5,
    data_dir="/home/v-oostapenko/data/mmlu",
    save_dir="/home/v-oostapenko/results/mmlu/llama-7B/",
    model_name="",
    model_path="",
    # tokenizer_name_or_path,
    openai_engine=None,
    subjects=-1,
    n_instances=None,
    eval_batch_size=1,
    skill_selector="topic",
    amlt_experiment_name="alpaca_smear",
    use_chat_format=False,
):
    return eval_mlu(
        ntrain,
        data_dir,
        save_dir,
        model_name,
        model_path,
        openai_engine,
        subjects,
        n_instances,
        eval_batch_size,
        skill_selector,
        amlt_experiment_name,
        use_chat_format,
    )


def eval_mlu(
    ntrain=5,
    data_dir="~/data/mmlu",
    save_dir="~/out/mmlu/llama-7B/",
    model_name="",
    model_path="",
    # tokenizer_name_or_path,
    openai_engine=None,
    subjects=-1,
    n_instances=None,
    eval_batch_size=1,
    skill_selector="topic",
    amlt_experiment_name="alpaca_smear",
    use_chat_format=False,
):
    from_hf = 0  # TODO: make sure this can also eval model from hf
    ################################################################################################
    # set paths
    code_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    save_dir = os.getenv("AMLT_OUTPUT_DIR", save_dir)
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:  # on gcr
        data_dir = "/mnt/default/data/mmlu/"
        base_model_path = "/mnt/default/data/models"
        base_cluster_infos = "/mnt/default/data/"  # /mnt/amlt_code/inst_follow/"
    else:
        base_cluster_infos = code_dir
    path_to_clusterer = f"{base_cluster_infos}/cluster_infos/cbtm/"
    ################################################################################################

    if model_path == "" and from_hf == 0:
        if model_name in model_dict:
            # can also list models in the model dict
            from_hf = model_dict[model_name]["from_hf"]
            model_path = model_dict[model_name]["model_path"]
            out_prefix = model_name
        elif not from_hf:
            exp_name = amlt_experiment_name
            from_hf = 0
            model_path = glob.glob(
                f"{base_model_path}/{exp_name}/{model_name}/yahma*/loss=*.ckpt"
            )[0]
            out_prefix = model_name

    # pack all the arguments into dict
    args = {
        "ntrain": ntrain,
        "data_dir": data_dir,
        "save_dir": save_dir,
        "model_name": model_name,
        "model_path": model_path,
        "openai_engine": openai_engine,
        "subjects": subjects,
        "n_instances": n_instances,
        "eval_batch_size": eval_batch_size,
        "skill_selector": skill_selector,
        "amlt_experiment_name": amlt_experiment_name,
        "use_chat_format": use_chat_format,
        "from_hf": from_hf,
    }
    args = dict_to_dataclass(args)

    base_model = "yahma/llama-7b-hf"
    model, tokenizer, config, topic_router = load_model_for_generation(
        from_hf, base_model, model_name, model_path, skill_selector, code_dir=code_dir
    )

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if subjects:
        assert all(
            subj in subjects for subj in subjects
        ), f"Some of the subjects you specified are not valid: {subjects}"
        subjects = subjects

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[:ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        if n_instances and n_instances < test_df.shape[0]:
            test_df = test_df.sample(n_instances, random_state=42)

        # if mode:
        # if skill_selector == "topic":
        #     cors, acc, probs = eval_hf_model(
        #         args,
        #         subject,
        #         model,
        #         tokenizer,
        #         dev_df,
        #         test_df,
        #         eval_batch_size,
        #         topic_router=topic_router,
        #         skill_selector=skill_selector,
        #         cluster_depth=cluster_depth,
        #     )
        # else:

        cors, acc, probs = eval_hf_model(
            args,
            subject,
            model,
            tokenizer,
            dev_df,
            test_df,
            eval_batch_size,
        )
        # else:
        #     cors, acc, probs = eval_openai_chat_engine(
        #         args, subject, openai_engine, dev_df, test_df, eval_batch_size
        #     )

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(save_dir, "{}.csv".format(subject)),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat])) for cat in cat_cors
                },
            },
            f,
        )
    return weighted_acc


if __name__ == "__main__":
    main()
