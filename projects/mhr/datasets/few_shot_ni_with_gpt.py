import json
import os
import numpy as np
import click
import json
import tqdm

from mttl.dataloader import ni_metrics
from mttl.models import gpt_utils


test_tasks = [
    "task1159_bard_analogical_reasoning_containers",
    "task957_e2e_nlg_text_generation_generate",
    "task1728_web_nlg_data_to_text",
    "task619_ohsumed_abstract_title_generation",
    "task034_winogrande_question_modification_object",
    "task1664_winobias_text_generation",
    "task1394_meta_woz_task_classification",
    "task738_perspectrum_classification",
    "task1615_sick_tclassify_b_relation_a",
    "task936_defeasible_nli_snli_classification",
    "task1155_bard_analogical_reasoning_trash_or_treasure",
    "task620_ohsumed_medical_subject_headings_answer_generation",
    "task890_gcwd_classification",
    "task362_spolin_yesand_prompt_response_sub_classification",
    "task304_numeric_fused_head_resolution",
    "task1540_parsed_pdfs_summarization",
    "task648_answer_generation",
    "task1516_imppres_naturallanguageinference",
    "task1161_coda19_title_generation",
    "task614_glucose_cause_event_detection",
]


def load_instructions(path):
    """
    Read all .json files in the directory path and
    extract the field "Definition" from each file.
    """
    import glob

    for file in glob.glob(path + "/*.json"):
        with open(file) as f:
            data = json.load(f)
            task_name = os.path.basename(file).replace(".json", "")
            if task_name in test_tasks:
                if data["train_examples"]:
                    yield task_name, data["train_examples"][0]["Definition"], data["train_examples"], data["test_examples"]


@click.command()
@click.option("--data_path", type=str)
@click.option("--model_name", type=str, default="gpt3")
@click.option("--model_type", type=str, default="text-davinci-003")
def main(data_path, model_name="gpt3", model_type="text-davinci-003"):
    task_results = {}
    gpt_utils.forward_instantiate(model_name=model_name, model_type=model_type)

    for nshot in [0, 1, 5, 10, 15, 20]:
        for inst in load_instructions(data_path):
            task_name, definition, train_exs, test_exs = inst
            batch = []
            outputs = []

            task_results[task_name] = ([], [])
            train_examples = [ex for ex in train_exs[:nshot]]

            for example in tqdm.tqdm(test_exs):
                if nshot == 0:
                    batch.append(definition[0] + "\n" + example["Instance"]['input'] + "\n" + "The answer is:\n")
                else:
                    msg = definition[0] + "\n"
                    for i in range(nshot):
                        if len(msg.split()) >= 1000:
                            print(i)
                            break
                        msg += train_examples[i]['Instance']['input'] + "\n" + "The answer is:\n" + train_examples[i]['Instance']['output'][0] + "\n\n"
                    msg += example["Instance"]['input'] + "\n" + "The answer is:\n"
                    batch.append(msg)

                outputs.append(example['Instance']['output'])
                if len(batch) == 10:
                    gens = gpt_utils.forward_evaluate(batch, async_generation=True, temperature=0.7)

                    task_results[task_name][0].extend(gens)
                    task_results[task_name][1].extend(outputs)

                    batch = []
                    outputs = []

            if len(batch):
                gens = gpt_utils.forward_evaluate(batch, temperature=0.7)
                task_results[task_name][0].extend(gens)
                task_results[task_name][1].extend(outputs)
                batch = []
                outputs = []

            rougeL = ni_metrics.compute_metrics(task_results[task_name][0], task_results[task_name][1])['rougeL']
            task_results[task_name] = rougeL
            print(task_name, rougeL)

        print("N-shot performance:", np.mean(list(task_results.values())))
        task_results["mean"] = np.mean(list(task_results.values()))

        with open("./ni-nshot{}.json".format(nshot), "w") as f:
            f.write(json.dumps(task_results))