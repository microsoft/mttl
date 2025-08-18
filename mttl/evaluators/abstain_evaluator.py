from tqdm.auto import tqdm
from mttl.evaluators.base import GenerativeEvaluator, switch_to_eval_mode
from mttl.logging import logger
from mttl.evaluators.metrics import compute_metrics


def answer_parsing(response):
    # mode 1: answer directly after
    temp = response.strip().split(" ")
    for option in ["A", "B", "C", "D", "E"]:
        if option in temp[0]:
            return option
    # mode 2: "The answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the answer is " + option in temp:
            return option.upper()
    # mode 3: "Answer: A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "answer: " + option in temp:
            return option.upper()
    # mode 4: " A/B/C/D/E " or " A/B/C/D/E."
    for option in ["A", "B", "C", "D", "E"]:
        if " " + option + " " in response or " " + option + "." in response:
            return option
    # mode 5: "The correct answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the correct answer is " + option in temp:
            return option.upper()
    # mode 6: "A: " or "B: " or "C: " or "D: " or "E: "
    for option in ["A", "B", "C", "D", "E"]:
        if option + ": " in response:
            return option
    # mode 7: "A/B/C/D/E" and EOS
    try:
        for option in ["A", "B", "C", "D", "E"]:
            if option + "\n" in response or response[-1] == option:
                return option
    except:
        pass
    # fail to parse
    print("fail to parse answer", response, "------------------")
    return "Z" # so that its absolutely wrong


class AbstainQAEvaluator(GenerativeEvaluator):
    def __init__(self, datamodule, use_vllm=False, generation_kwargs=None):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
        return_predictions=False,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        abstain_flags = []
        correct_flags = []
        abstain_scores = None
        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            predictions = self.generate_for_batch(model, batch).generated_texts
            for i, (pred_raw, source, target_raw) in enumerate(
                zip(predictions, batch["sources_texts"], batch["labels_texts"])
            ):
                pred = answer_parsing(pred_raw)
                if "[/INST]" in target_raw:
                    target = answer_parsing(target_raw.split("[/INST]")[1].split("</s>")[0])
                else:
                    target = target_raw
                if "sorry" in pred.lower():
                    abstain_flags.append(1)
                else:
                    abstain_flags.append(0)
                logger.info(f"Pred: {pred}, Target: {target}")
                if pred == target:
                    correct_flags.append(1)
                else:
                    correct_flags.append(0)

        metrics = compute_metrics(correct_flags, abstain_flags, abstain_scores)
        print(metrics)
        return metrics
