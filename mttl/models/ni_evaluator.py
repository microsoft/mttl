from collections import defaultdict
from copy import deepcopy
from mttl.dataloader.ni_metrics import compute_metrics
import tqdm


class NIEvaluator(object):
    def __init__(self, config, data_dir=None, num_pos_examples=0, device="cuda"):
        from mttl.datamodule.ni_original_data_module import NIOriginalDataModule

        self.config = deepcopy(config)
        self.device = device
        self.config.num_pos_examples = num_pos_examples

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = data_dir
        self.datamodule = NIOriginalDataModule(self.config, for_generation=True)
        self.datamodule.setup("test")

    def evaluate(self, model, metric_per_task=True):
        tokenizer = self.datamodule.tokenizer
        samples_seen = 0

        # DDP
        if hasattr(model, "module"):
            model = model.module

        def decode(preds):
            preds[preds == -100] = tokenizer.pad_token_id
            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            return preds

        all_predictions = []
        all_references = []
        task_names = []

        dataloader = self.datamodule.test_dataloader()
        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for step, batch in pbar:
            task_name = batch.pop("task_names", None)
            texts = batch.pop("input_texts", None)
            batch.pop("labels_texts", None)

            max_length = self.config.max_output_length
            max_length += batch["input_ids"].shape[-1]

            batch["input_ids"] = batch["input_ids"].to(self.device)
            predictions = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["input_ids"].ne(tokenizer.pad_token_id),
                max_length=max_length,
                generation_config=model.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            predictions = predictions.sequences
            predictions = predictions[:, batch["input_ids"].shape[-1] :]
            predictions = decode(predictions)
            references = decode(batch["labels"])

            # If we are in a multiprocess environment, the last batch has duplicates
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
                task_name = task_name[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += len(references)

            all_predictions += predictions
            all_references += references
            task_names += task_name

            eval_metrics = compute_metrics(
                all_predictions, [[r] for r in all_references], reduction="mean"
            )
            pbar.set_description("rougeL: {:.4f}".format(eval_metrics["rougeL"]))

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        mean_metrics = {}
        for metric_name, metric_value in eval_metrics.items():
            metric_value = sum(eval_metrics[metric_name]) / len(eval_metrics[metric_name])
            mean_metrics[metric_name] = metric_value

        if metric_per_task:
            for metric_name in eval_metrics.keys():
                metric_values = defaultdict(list)
                for task_name, v in zip(task_names, eval_metrics[metric_name]):
                    metric_values[task_name] += [v]
            metric_values = {
                task_name: sum(vs) / len(vs)
                for task_name, vs in metric_values.items()
            }
            metric_values["all"] = mean_metrics[metric_name]
        else:
            metric_values = mean_metrics
        return metric_values
