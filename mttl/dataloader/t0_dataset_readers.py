import os
import json
import numpy as np
from datasets import load_from_disk
import pkg_resources
from promptsource import templates
from promptsource.templates import DatasetTemplates
import csv
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd

from mttl.models.modifiers.expert_containers.expert_library import DatasetLibrary


DATASETS_OFFLINE = "no"
MAX_EXAMPLES_PER_DATASET = 500_000
TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # Tasks with broken cached files
    "gigaword_summarize_",
]


DATASET_INFO = """HF_name,subset,task_by_convention,do_train,do_eval,train_size
crows_pairs,,bias_and_fairness,,BIAS_FAIRNESS,
jigsaw_toxicity_pred,,bias_and_fairness,,BIAS_FAIRNESS,
super_glue,axg,bias_and_fairness,,BIAS_FAIRNESS,
wino_bias,type1_anti,bias_and_fairness,,BIAS_FAIRNESS,
wino_bias,type2_anti,bias_and_fairness,,BIAS_FAIRNESS,
wino_bias,type1_pro,bias_and_fairness,,BIAS_FAIRNESS,
wino_bias,type2_pro,bias_and_fairness,,BIAS_FAIRNESS,
super_glue,wsc.fixed,coreference,SGLUE,BASE,554
winogrande,winogrande_xl,coreference,,BASE,40398
super_glue,cb,NLI,,BASE,250
super_glue,rte,NLI,,BASE,2490
anli,,NLI,,BASE,162865
glue,mrpc,paraphrase,BASE,,3668
glue,qqp,paraphrase,BASE,,363846
paws,labeled_final,paraphrase,BASE,,49401
ai2_arc,ARC-Challenge,QA_closed_book,GPT_EVAL,,1119
ai2_arc,ARC-Easy,QA_closed_book,GPT_EVAL,,2251
kilt_tasks,hotpotqa,QA_closed_book,BASE,,88869
trivia_qa,unfiltered,QA_closed_book,GPT_EVAL,,87622
web_questions,,QA_closed_book,GPT_EVAL,,3778
wiki_qa,,QA_closed_book,BASE,,20360
adversarial_qa,dbidaf,QA_extractive,BASE,,10000
adversarial_qa,dbert,QA_extractive,BASE,,10000
adversarial_qa,droberta,QA_extractive,BASE,,10000
duorc,SelfRC,QA_extractive,BASE,,60721
duorc,ParaphraseRC,QA_extractive,BASE,,69524
ropes,,QA_extractive,BASE,,10924
squad_v2,,QA_extractive,GPT_EVAL,,130319
super_glue,record,QA_extractive,SGLUE,,100730
quoref,,QA_extractive,BASE,,19399
cos_e,v1.11,QA_multiple_choice,BASE,,9741
cosmos_qa,,QA_multiple_choice,BASE,,25262
dream,,QA_multiple_choice,BASE,,6116
openbookqa,main,QA_multiple_choice,GPT_EVAL,,4957
qasc,,QA_multiple_choice,BASE,,8134
quail,,QA_multiple_choice,BASE,,10246
quarel,,QA_multiple_choice,BASE,,1941
quartz,,QA_multiple_choice,BASE,,2696
race,high,QA_multiple_choice,GPT_EVAL,,62445
race,middle,QA_multiple_choice,GPT_EVAL,,25421
sciq,,QA_multiple_choice,BASE,,11679
social_i_qa,,QA_multiple_choice,BASE,,33410
super_glue,boolq,QA_multiple_choice,SGLUE,,9427
super_glue,copa,QA_multiple_choice,SGLUE,BASE,400
super_glue,multirc,QA_multiple_choice,SGLUE,,27243
wiki_hop,original,QA_multiple_choice,BASE,,43738
wiqa,,QA_multiple_choice,BASE,,29808
piqa,,QA_multiple_choice,GPT_EVAL,,16113
amazon_polarity,,sentiment,BASE,,3600000
app_reviews,,sentiment,BASE,,288065
imdb,,sentiment,BASE,,25000
rotten_tomatoes,,sentiment,BASE,,8530
yelp_review_full,,sentiment,BASE,,650000
story_cloze,2016,story_completion,,BASE,
hellaswag,,story_completion,GPT_EVAL,BASE,39905
common_gen,,structure_to_text,BASE,,67389
wiki_bio,,structure_to_text,BASE,,582659
cnn_dailymail,3.0.0,summarization,BASE,,287113
gigaword,,summarization,BASE,,3803957
multi_news,,summarization,BASE,,44972
samsum,,summarization,BASE,,14732
xsum,,summarization,BASE,,204045
ag_news,,topic_classification,BASE,,120000
dbpedia_14,,topic_classification,BASE,,560000
trec,,topic_classification,BASE,,5452
super_glue,wic,word_sense_disambiguation,SGLUE,BASE,5428
"""

# num shots for test tasks
NUM_SHOTS_CONFIG = {
    "wsc": 32,
    "wic": 32,
    "winogrande": 50,
    "storycloze": 70,
    "rte": 32,
    "cb": 32,
    "copa": 32,
    "h-swag": 20,
    "anli-r1": 32,
    "anli-r2": 32,
    "anli-r3": 32,
}


class T0DatasetConfig:
    def __init__(self, dataset, use_t0_templates_as_tasks, seed):
        self.dataset = dataset
        self.data_dir = os.environ["T0_DATA_DIR"]
        self.num_shot = NUM_SHOTS_CONFIG.get(dataset, 32)
        self.few_shot_random_seed = seed
        self.train_template_idx = -1
        self.eval_template_idx = -1
        self.change_hswag_templates = False
        self.raft_cross_validation = True
        self.use_t0_templates_as_tasks = use_t0_templates_as_tasks
        self.raft_validation_start = 0
        self.raft_labels_in_input_string = "comma"
        self.cleaned_answer_choices_b77 = False

    @classmethod
    def from_args(cls, args):
        config = T0DatasetConfig(
            args.finetune_task_name,
            args.use_t0_templates_as_tasks,
            args.seed,
        )
        return config


class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash
        self.metric = "accuracy"

        self.templates = DatasetTemplates(*self.dataset_stash)
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)

    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx == -1:
            list_idx = []
            list_templates = []
            for idx, template_name in enumerate(template_names):
                if self.templates[template_name].metadata.original_task:
                    list_idx.append(idx)
                    list_templates.append(self.templates[template_name])
            print(list_idx)

            return list_templates
        elif template_idx == -2:
            return [self.templates[template_name] for template_name in template_names]

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(
                os.path.join(DATASETS_OFFLINE, *self.dataset_stash)
            )[split]
        else:
            orig_data = DatasetLibrary.pull_dataset(
                *self.dataset_stash, split=split, cache_dir=self.config.data_dir
            )
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = pkg_resources.resource_filename(
            __name__,
            f"{self.config.data_dir}/few_shot/{self.config.dataset}/{self.config.num_shot}_shot",
        )
        file_path = os.path.join(
            file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl"
        )

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))
            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            with open(file_path, "w+") as fout:
                for example in selected_data:
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: self.config.num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def compute_metric(self, accumulated):
        matching = [
            a == b for a, b in zip(accumulated["prediction"], accumulated["label"])
        ]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


class StoryClozeReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("story_cloze", "2016"))

    def read_orig_dataset(self, split):
        if split == "train":
            split = "validation"
        elif split == "validation":
            split = "test"

        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(
                os.path.join(DATASETS_OFFLINE, *self.dataset_stash)
            )[split]
        else:
            dataset_stash = ("story_cloze", "2016")
            orig_data = DatasetLibrary.pull_dataset(
                *dataset_stash,
                split=split,
                data_dir=os.environ.get("STORYCLOZE_DIR", self.config.data_dir),
            )
        orig_data = [example for example in orig_data]
        for idx, example in enumerate(orig_data):
            example["label"] = example["answer_right_ending"] - 1
            example["idx"] = idx
        return orig_data


class ANLIR1Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r1")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR2Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r2")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR3Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r3")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class WSCFixedReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wsc.fixed"))


class RTEReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "rte"))


class HSwagReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("hellaswag",))
        if config.change_hswag_templates:
            from promptsource.templates import Template

            name_jinja = [
                ("basic", "{{ctx}}|||{{endings [label | int()]}}"),
                (
                    "prompt 1",
                    "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "prompt 2",
                    "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "prompt 3",
                    "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "prompt 4",
                    "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "ctx a,b",
                    "Complete the description with an appropriate ending:\n First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }} ...|||{{answer_choices [label | int()]}}",
                ),
                (
                    "middle",
                    "If a description of a situation begins like this: {{ ctx }}... Then how does it continue?|||{{answer_choices [label | int()]}}",
                ),
            ]

            self.templates = []
            for name, jinja in name_jinja:
                self.templates.append(
                    Template(
                        name=name,
                        jinja=jinja,
                        reference="",
                        answer_choices='{{endings | join("|||")}}',
                    )
                )

            if self.config.train_template_idx >= 0:
                self.train_template = self.templates[self.config.train_template_idx]
            else:
                self.train_template = self.templates
            if self.config.eval_template_idx >= 0:
                self.eval_template = self.templates[self.config.eval_template_idx]
            else:
                self.eval_template = self.templates

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        for idx, example in enumerate(orig_data):
            example["label"] = int(example["label"])
            example["idx"] = idx
        return orig_data


class WiCReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wic"))


class COPAReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "copa"))

    def get_template(self, template_idx):
        if template_idx >= 0:
            return super().get_template(template_idx)
        else:
            return super().get_template(template_idx)[:8]


class WinograndeReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("winogrande", "winogrande_xl"))

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        for idx, example in enumerate(orig_data):
            example["label"] = int(example["answer"]) - 1
            example["idx"] = idx
        return orig_data


class CBReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "cb"))


class T0MixtureReader(object):
    def __init__(self, config):
        self.config = config

        datatset_subset_tuple = Tuple[str, Optional[str]]
        t0_train: Dict[str, List[datatset_subset_tuple]] = {
            "BASE": [],
            # GPT3 evaluation set
            "GPT_EVAL": [],
            # SuperGLUE (except RTE and CB)
            "SGLUE": [],
        }
        t0_eval: Dict[str, List[datatset_subset_tuple]] = {
            "BASE": [],
            "BIAS_FAIRNESS": [],
        }
        gsheet: Dict[datatset_subset_tuple, Dict] = {}

        reader = csv.DictReader(DATASET_INFO.splitlines())
        for row in reader:
            if row["subset"] == "":
                row["subset"] = None  # to match promptsource.Template object
            dataset_subset = (row["HF_name"], row["subset"])
            if row["do_train"] != "":
                do_train_source = row["do_train"]
                # sanity checks
                if do_train_source == "SGLUE":
                    assert dataset_subset[0] == "super_glue"
                t0_train[do_train_source].append(dataset_subset)
            if row["do_eval"] != "":
                do_eval_source = row["do_eval"]
                # sanity checks
                if do_eval_source == "BIAS_FAIRNESS":
                    assert row["task_by_convention"] == "bias_and_fairness"
                t0_eval[do_eval_source].append(dataset_subset)
            gsheet[dataset_subset] = row

        all_datasets = sum(t0_train.values(), []) + sum(t0_eval.values(), [])
        all_templates = templates.TemplateCollection()
        all_templates.remove("anli")

        # 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
        t0_train_mixture: Dict[str, List[str]] = {key: [] for key in t0_train}
        t0_eval_mixture: Dict[str, List[str]] = {key: [] for key in t0_eval}
        mixture_cap: Dict[str, int] = {}
        single_original_task: Dict[Tuple[str, str], str] = {}
        all_original_tasks: List[str] = []
        added_tasks: List[Tuple[str, str, str]] = []

        def get_task_name(dataset_name, subset_name, template_name):
            # Clean the text according to allowed characters for a task name
            task_name = (
                dataset_name
                + (f"_{subset_name}_" if subset_name is not None else "_")
                + template_name
            )
            return re.sub(r"[^\w\d\._]+", "_", task_name)

        for dataset_name, subset_name in all_templates.keys:
            if (dataset_name, subset_name) not in all_datasets:
                all_templates.remove(dataset_name, subset_name)
                continue

            dataset = all_templates.get_dataset(dataset_name, subset_name)
            num_templates = len(dataset.all_template_names)
            train_size = gsheet[(dataset_name, subset_name)]["train_size"]
            if train_size == "":
                train_size = 0
            else:
                train_size = int(train_size)
            if train_size > MAX_EXAMPLES_PER_DATASET // num_templates:
                cap = MAX_EXAMPLES_PER_DATASET // num_templates
            else:
                cap = train_size
            for template_name in dataset.all_template_names:
                added_tasks.append((dataset_name, subset_name, template_name))

                template = dataset[template_name]

                task_name = get_task_name(dataset_name, subset_name, template_name)

                if (
                    dataset_name,
                    subset_name,
                ) not in single_original_task and template.metadata.original_task:
                    single_original_task[(dataset_name, subset_name)] = task_name

                if template.metadata.original_task:
                    all_original_tasks.append(task_name)

                # Check that the dataset_subset_tuple is in t0_train
                for key, dataset_subset_tuples in t0_train.items():
                    if (dataset_name, subset_name) in dataset_subset_tuples:
                        t0_train_mixture[key].append(task_name)
                        mixture_cap[task_name] = cap

                # Check that the dataset_subset_tuple is in t0_eval
                if (dataset_name, subset_name) in t0_eval["BASE"]:
                    if template.metadata.original_task:
                        t0_eval_mixture["BASE"].append(task_name)

                # TODO use template.metadata.answer_choices here for rank eval
                if (dataset_name, subset_name) in t0_eval["BIAS_FAIRNESS"]:
                    t0_eval_mixture["BIAS_FAIRNESS"].append(task_name)

        self.t0_base_tasks = []
        self.t0_base_templates = []

        for dataset_name, subset_name, template_name in added_tasks:
            task_name = get_task_name(dataset_name, subset_name, template_name)

            if task_name in t0_train_mixture["BASE"]:
                if task_name not in TASK_BLACKLIST:
                    self.t0_base_tasks.append(
                        (
                            dataset_name,
                            subset_name,
                            template_name,
                            mixture_cap[task_name],
                        )
                    )
                    template = all_templates.get_dataset(dataset_name, subset_name)[
                        template_name
                    ]
                    self.t0_base_templates.append(template)

        print("#", len(self.t0_base_tasks), "tasks and templates loaded.")
        print("#", len(self.get_base_tasks()), "base tasks loaded.")
        print(self.get_base_tasks())

    def get_base_tasks(self):
        if self.config.use_t0_templates_as_tasks:
            return sorted(
                set([f"{data[0]}/{data[1]}/{data[2]}" for data in self.t0_base_tasks])
            )
        else:
            return sorted(set([f"{data[0]}/{data[1]}" for data in self.t0_base_tasks]))

    def get_template(self):
        return self.t0_base_templates

    def read_orig_dataset(self, split):
        """
        Read the original dataset
        :param split: split of data
        """
        import tqdm

        orig_data = []
        for dataset_name, subset_name, template_name, cap in tqdm.tqdm(
            self.t0_base_tasks, desc="Loading data..."
        ):
            if split == "train":
                split_num = f"{split}[0:{cap}]"
            else:
                split_num = split

            ds = DatasetLibrary.pull_dataset(
                dataset_name,
                subset_name,
                split=split_num,
                cache_dir=self.config.data_dir,
            )
            ds.dataset_name = dataset_name
            ds.subset_name = subset_name
            ds.template_name = template_name
            orig_data.append(ds)

        return orig_data


class RaftTemplate(object):
    def __init__(self, config, answer_choices):
        with open(
            os.path.join(config.data_dir, "raft_prompt_construction_settings.jsonl")
        ) as f:
            data = [json.loads(line) for line in f]
            FIELD_ORDERING = data[0]
            INSTRUCTIONS = data[1]
        self.dataset_name = config.dataset
        self.answer_choices = answer_choices
        self.instruction = INSTRUCTIONS[self.dataset_name]
        self.fields = FIELD_ORDERING[self.dataset_name]
        self.raft_labels_in_input_string = config.raft_labels_in_input_string

    def apply(self, example):
        if self.raft_labels_in_input_string == "comma":
            input_str = [
                self.instruction.strip()
                + " Possible labels: "
                + ", ".join(
                    [choice for index, choice in enumerate(self.answer_choices)]
                )
            ]
        elif self.raft_labels_in_input_string == "newline":
            input_str = [
                self.instruction.strip()
                + "\nPossible labels:\n"
                + "\n".join(
                    [
                        str(index + 1) + ". " + choice
                        for index, choice in enumerate(self.answer_choices)
                    ]
                )
            ]
        else:
            input_str = [self.instruction.strip()]

        for key in example:
            if key in self.fields:
                if example[key].strip() != "":
                    input_str.append(str(key) + ": " + example[key].strip())

        if example["label"] == -1:
            target_str = "Unlabeled"
        else:
            target_str = self.answer_choices[example["label"]]
        input_str[-1] += "\nLabel:"
        return input_str, target_str

    def get_answer_choices_list(self, example):
        return self.answer_choices


class RaftReader(object):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset
        self.orig_data = DatasetLibrary.pull_dataset(
            "ought/raft", name=self.dataset_name
        )
        self.answer_choices = self.orig_data["train"].features["Label"].names[1:]

        if self.config.dataset == "banking_77" and config.cleaned_answer_choices_b77:
            self.answer_choices = [
                answer.replace("_", " ").replace(". ", " ")
                for answer in self.answer_choices
            ]

        self.template = RaftTemplate(config, self.answer_choices)

    def get_train_template(self):
        return self.template

    def get_eval_template(self):
        return self.template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if self.config.raft_cross_validation:
            orig_data = [example for example in self.orig_data["train"]]
            if split == "train":
                orig_data = (
                    orig_data[: self.config.raft_validation_start]
                    + orig_data[self.config.raft_validation_start + 10 :]
                )
                assert len(orig_data) == 40
            elif split == "validation":
                orig_data = orig_data[
                    self.config.raft_validation_start : self.config.raft_validation_start
                    + 10
                ]
                assert len(orig_data) == 10
        else:
            if split == "validation":
                split = "test"
            orig_data = [example for example in self.orig_data[split]]
        for i, example in enumerate(orig_data):
            example["label"] = int(example["Label"]) - 1
            example["idx"] = example["ID"]
        return orig_data

    def compute_metric(self, accumulated):
        data = []
        idxs = accumulated["idx"]
        predictions = accumulated["prediction"]
        for idx, prediction in zip(idxs, predictions):
            data.append({"ID": idx, "Label": self.answer_choices[prediction]})
        result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype(
            {"ID": int, "Label": str}
        )
        result_df.to_csv(self.config.dev_pred_file, index=False)
        matching = [
            a == b for a, b in zip(accumulated["prediction"], accumulated["label"])
        ]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


def get_dataset_reader(args):
    dataset_class = {
        "T0Mixture": T0MixtureReader,
        "rte": RTEReader,
        "h-swag": HSwagReader,
        "copa": COPAReader,
        "wic": WiCReader,
        "winogrande": WinograndeReader,
        "cb": CBReader,
        "storycloze": StoryClozeReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "wsc": WSCFixedReader,
        "ade_corpus_v2": RaftReader,
        "banking_77": RaftReader,
        "terms_of_service": RaftReader,
        "tai_safety_research": RaftReader,
        "neurips_impact_statement_risks": RaftReader,
        "overruling": RaftReader,
        "systematic_review_inclusion": RaftReader,
        "one_stop_english": RaftReader,
        "tweet_eval_hate": RaftReader,
        "twitter_complaints": RaftReader,
        "semiconductor_org_types": RaftReader,
    }

    if args.finetune_task_name is None:
        args.finetune_task_name = "T0Mixture"

    dataset_class = dataset_class[args.finetune_task_name]

    return dataset_class(T0DatasetConfig.from_args(args))
