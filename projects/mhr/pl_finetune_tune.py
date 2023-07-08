from dataclasses import dataclass
from pl_finetune import finetune_ni, finetune_xfit
from mttl.config import parse_config


@dataclass
class NI_HPS:
    LRS = [1e-3, 1e-4]
    MLRS = [0.5, 0.1]


def finetune_hps(args):
    import os
    import pickle
    import copy

    best_args = None
    best_metric_perf = float("-inf")
    best_metrics = None

    trials = []

    if args.finetune_type == "Z":
        for mlr in NI_HPS.MLRS:
            trials.append({
                "module_logits_learning_rate": mlr
            })
    elif args.model_modifier in ["poly_lora", "poly_ia3"]:
        for lr in NI_HPS.LRS:
            for mlr in NI_HPS.MLRS:
                trials.append({
                    "learning_rate": lr,
                    "module_logits_learning_rate": mlr
                })
    else:
        for lr in NI_HPS.LRS:
            trials.append({
                "learning_rate": lr,
            })

    for trial in trials:
        c_args = copy.deepcopy(args)
        
        for name, val in trial.items():
            setattr(c_args, name, val)

        print("Trying...", trial)

        if c_args.dataset == "ni":
            df = finetune_ni(c_args, use_mlf=False, do_zs=False)
        else:
            df = finetune_xfit(c_args, use_mlf=False, do_zs=False)

        if df['val/metric_perf'].mean() > best_metric_perf:
            best_metric_perf = df['val/metric_perf'].mean()
            best_args = c_args
            best_metrics = df

    best_metrics.to_csv(os.path.join(args.output_dir, "result.csv"))

    with open(os.path.join(args.output_dir, "best_args.pkl"), "wb") as f:
        pickle.dump(best_args, f)


if __name__ == "__main__":
    config = parse_config()

    finetune_hps(config)
