import re 
import torch.optim as optim
from collections import defaultdict
from transformers import Adafactor

from mttl.utils import logger


def get_optimizer(model, args, no_decay=None):
    """
    Construct optimizer based on args

    :param model:
    :param args:
    :return:
    """
    optim_name = args.optimizer
    no_decay = [] if no_decay is None else no_decay

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()

    for (param_name, param) in model.named_parameters():     
        if re.fullmatch(args.trainable_param_names, param_name) and (
            not args.non_trainable_param_names
            or not re.fullmatch(args.non_trainable_param_names, param_name)
        ):
            if any(nd in param_name for nd in no_decay):
                param_groups["no_decay"]["params"].append(param)
            elif "module_logits" in param_name:   
                param_groups["module_logits"]["params"].append(param)
            elif "lora" in param_name:
                param_groups["adapters"]["params"].append(param)
            elif "router" in param_name:
                param_groups["router"]["params"].append(param)
            else:
                param_groups["others"]["params"].append(param)
            trainable_param_names.add(param_name)
        else:
            param.requires_grad = False

    for name in sorted(trainable_param_names):
        logger.info("Training parameter: %s", name)

    for key in param_groups.keys():
        if key in ["module_logits"]:
            param_groups[key]["lr"] = (
                args.module_logits_learning_rate
                if args.module_logits_learning_rate is not None
                else args.learning_rate
            )
            logger.info("Module logits learning rate: %s", param_groups[key]["lr"])
        elif key in ["adapters"]:
            param_groups[key]["weight_decay"] = (
                args.adapters_weight_decay
                if args.adapters_weight_decay is not None
                else args.weight_decay
            )
            param_groups[key]["lr"] = ( 
                args.adapters_learning_rate  
                if key in ["adapters"] and args.adapters_learning_rate
                else args.learning_rate
            )
            logger.info("Adapters learning rate: %s", param_groups[key]["lr"])
        elif key in ["router"]:          
            param_groups[key]["weight_decay"] = (
                args.router_weight_decay
                if args.router_weight_decay is not None
                else args.weight_decay
            )
            param_groups[key]["lr"] = (
                args.router_learning_rate
                if key in ["router"] and args.router_learning_rate
                else args.learning_rate
            )
            logger.info("Router learning rate: %s", param_groups[key]["lr"])
        else:  
            param_groups[key]["weight_decay"] = (
                0.0 if key in ["module_logits", "no_decay"] else args.weight_decay
            )
            param_groups[key]["lr"] = args.learning_rate

    param_groups = param_groups.values()
    if optim_name.lower() == "adam":   
        optimizer = optim.Adam(param_groups)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups)
    elif optim_name.lower() == "adamw":
        # from transformers import AdamW # tloen uses adamw_torch
        from torch.optim import AdamW
        optimizer = AdamW(param_groups, eps=args.adam_epsilon)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            scale_parameter=args.adafactor_scale_parameter,
            relative_step=args.adafactor_relative_step,
            warmup_init=args.adafactor_warmup_init,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return optimizer, trainable_param_names
