import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--train_dir", default="data", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str, help="wandb experience name")
    parser.add_argument("--wandb_project", type=str, default="poly-s")

    ## Model parameters
    parser.add_argument("--model", default="facebook/bart-large", required=False)
    parser.add_argument("--precision", type=str, default=32)
    parser.add_argument("--dataset", type=str, default="xfit")
    parser.add_argument("--finetune_task_name", default=None, type=str)
    parser.add_argument("--finetune_switch_to_avg_modules", action='store_true', help='Set Z to average, 1/|S|.')
    parser.add_argument("--finetune_fix_skills", action='store_true', help='Fix skills parameters, only train Z.')

    parser.add_argument("--finetune_full_model", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)

    # pick the key components
    parser.add_argument("--selector", type=str, default="polytropon")
    parser.add_argument("--combinor", type=str, default="weighted_sum")
    parser.add_argument("--processor", type=str, default="lora")

    ## ni dataset flags
    parser.add_argument("--use_task_descriptions", action="store_true")
    parser.add_argument("--num_pos_examples", type=int, default=0)

    ## Basic parameters

    parser.add_argument("--n_skills", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=1)

    parser.add_argument("--finegrained", action="store_true")
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--do_lowercase", action="store_true", default=False)
    parser.add_argument("--freeze_embeds", action="store_true", default=False)


    ## Loading from checkpoint parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="Load model from checkpoint.")
    ## Continue training from this checkpoint
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="This is the checkpoint of the backbone, e.g. if we fine-tuned T5 on NI for example.")
    parser.add_argument(
        "--use_precomputed_task_embeddings", action="store_true", default=False
    )

    # Preprocessing/decoding-related parameters
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--append_another_bos", action="store_true", default=False)

    # Training-related parameters
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--predict_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--module_logits_learning_rate", default=0.1, type=float)

    # TODO: maybe use this instead of warmup_steps
    parser.add_argument(
        "--warmup_proportion",
        default=0.06,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=0.1, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, type=int, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=-1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=-1,
        type=int,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--total_steps",
        default=-1,
        type=int,
        help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for saving predictions"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use a subset of data for debugging"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--custom_tasks_splits",
        type=str,
        default="dataloader/custom_tasks_splits/random.json",
    )
    return parser
