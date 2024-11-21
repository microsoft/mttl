from mttl.datamodule.mt_seq_to_seq_module import (
    FlanConfig,
    FlanModule,
)

flan = FlanModule(
    FlanConfig(
        "sordonia/flan-debug-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
    )
)

for batch in flan.train_dataloader():
    print(batch)
    break
