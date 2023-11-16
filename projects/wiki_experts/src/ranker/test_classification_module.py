from projects.wiki_experts.src.ranker.classification_module import (
    ClassificationDataModule,
    ClassificationConfig,
)

from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig


data_module = ClassificationDataModule(
    ClassificationConfig(
        dataset="sordonia/flan-10k-flat", model="EleutherAI/gpt-neo-125m"
    )
)

# data_module = FlanModule(
#     FlanConfig(
#         dataset="sordonia/flan-10k-flat",
#         model="EleutherAI/gpt-neo-125m",
#         finetune_task_name="adversarial_qa_dbert_answer_the_following_q",
#     ),
#     for_generation=True,
# )

for batch in data_module.train_dataloader():
    print(batch)
    break
