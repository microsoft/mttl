from dataclasses import dataclass

from mttl.datamodule.base import DatasetConfig, DefaultCollator, DataModule
from mttl.models.library.dataset_library import DatasetLibrary
import ast

# @dataclass
# class OrcaDataModuleCollator(DefaultCollator):
#     def __call__(self, batch):
#         sources = []
#         labels = []
#         for item in batch:
#             if type(item["messages"]) == str:
#                 item["messages"] = ast.literal_eval(item["messages"])

#             sources.append("You are a helpful assistant.")
#             labels.append(
#                 self.tokenizer.apply_chat_template(item["messages"], tokenize=False)
#             )
#         output_batch = (
#             self.prepare_inputs_for_gpt_family(sources, labels)
#             if self.model_family == "gpt"
#             else self.prepare_inputs_for_seq2seq_family(sources, labels)
#         )

#         output_batch["sources_texts"] = sources
#         output_batch["labels_texts"] = labels

#         return output_batch


@dataclass
class OrcaDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        train_dataset = DatasetLibrary.pull_dataset_with_retry(
            "zhan1993/orca_sqs_dataset"
        )["train"]
        if self.config.finetune_task_name:
            train_dataset = train_dataset.filter(
                lambda example: example["task_name"] == self.config.finetune_task_name
            )
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset, 0.1
        )
        self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = DatasetConfig(
        model="microsoft/Phi-3-mini-4k-instruct", finetune_task_name="creative_content"
    )
    from transformers import AutoTokenizer

    # <|assistant|>
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    message = [
        {
            "content": "Create an outline for a presentation that includes five engaging and informative slides to test the audience's understanding of the concepts discussed in your talk, such as p-value functions, compatibility intervals, and the misconceptions about p-values. Each slide should present a multiple-choice question related to these topics. Prepare talking points that explain the correct answers and why they are correct, ensuring the explanations are clear and suitable for an educational presentation.",
            "role": "user",
        },
        {
            "content": "Title: Understanding P-Values and Statistical Inference\n\nSlide 1: Introduction to P-Values\n- Multiple-Choice Question: What does a p-value indicate in hypothesis testing?\n A) The probability that the null hypothesis is true\n B) The probability of observing the data, or something more extreme, if the null hypothesis is true\n C) The probability that the alternative hypothesis is true\n D) The likelihood of making a Type I error\n\nTalking Points:\n- The correct answer is B. The p-value is the probability of observing the data, or something more extreme, if the null hypothesis is true. It is not a direct statement about the truth of the null or alternative hypothesis (A and C are incorrect). While it is related to the risk of a Type I error, it is not the likelihood of making this error (D is incorrect), but rather a measure of evidence against the null hypothesis given the data.\n\nSlide 2: Understanding Compatibility Intervals\n- Multiple-Choice Question: What does a 95% compatibility interval (CI) for a parameter estimate indicate?\n A) 95% of the data falls within this interval\n B) There is a 95% chance that the interval contains the true parameter value\n C) If we were to repeat the experiment many times, 95% of the calculated intervals would contain the true parameter value\n D) The parameter has a 95% probability of being within this interval\n\nTalking Points:\n- The correct answer is C. A 95% compatibility interval means that if we were to repeat the experiment many times, we would expect 95% of the calculated intervals to contain the true parameter value. It does not mean that 95% of the data falls within this interval (A is incorrect), nor does it imply a probability statement about the parameter itself (B and D are incorrect), as frequentist intervals do not assign probabilities to fixed parameters.\n\nSlide 3: Misconceptions About P-Values\n- Multiple-Choice Question: Which of the following is a common misconception about p-values?\n A) A low p-value confirms the research hypothesis\n B) A high p-value proves the null hypothesis is true\n C) P-values can indicate the size of an effect\n D) All of the above\n\nTalking Points:\n- The correct answer is D. All of the options listed are common misconceptions about p-values. A low p-value does not confirm the research hypothesis (A is incorrect); it simply suggests that the data are unlikely under the null hypothesis. A high p-value does not prove the null hypothesis is true (B is incorrect); it suggests that the data are not sufficiently unusual under the null hypothesis. P-values do not provide information about the size or importance of an effect (C is incorrect); they only indicate how incompatible the data are with the null hypothesis.\n\nSlide 4: P-Value Functions\n- Multiple-Choice Question: What information does a p-value function provide that a single p-value does not?\n A) The probability that the null hypothesis is true for different parameter values\n B) The p-values for a range of different alternative hypotheses\n C) The likelihood of the data under various parameter values\n D) The confidence level of the compatibility interval\n\nTalking Points:\n- The correct answer is B. A p-value function provides the p-values for a range of different alternative hypotheses, not just a single point alternative. It does not give the probability that the null hypothesis is true for different parameter values (A is incorrect), nor does it directly provide the likelihood of the data (C is incorrect). The confidence level of the compatibility interval is a separate concept (D is incorrect).\n\nSlide 5: Interpreting P-Values and Compatibility Intervals Together\n- Multiple-Choice Question: How should p-values and compatibility intervals be used together to interpret statistical results?\n A) To determine the probability that the null hypothesis is true\n B) To assess the evidence against the null hypothesis and the range of parameter values that are compatible with the data\n C) To calculate the effect size and its significance\n D) To prove the research hypothesis\n\nTalking Points:\n- The correct answer is B. P-values and compatibility intervals should be used together to assess the evidence against the null hypothesis (provided by the p-value) and to understand the range of parameter values that are compatible with the data (provided by the compatibility interval). They do not determine the probability that the null hypothesis is true (A is incorrect), nor do they prove the research hypothesis (D is incorrect). While they can help contextualize the effect size, they do not calculate it (C is incorrect).",
            "role": "assistant",
        },
    ]
    output = tokenizer.apply_chat_template(message, tokenize=False)

    # dataset = DatasetLibrary.pull_dataset_with_retry(
    #     "zhan1993/orca_sqs_dataset"
    # )

    # dataset = dataset.filter(lambda example: example["task_name"] == config.finetune_task_name)
    # splited_dataset = dataset.map(get_source_target,remove_columns=["messages","split"])
    # breakpoint()

    datamodule = OrcaDataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
