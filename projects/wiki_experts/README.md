# Wiki Expert

The `wiki-experts` library enables users to 

1. train different kinds of adapters (LoRA, soft prompts, …) over LLMs
2. build MoE models from these adapters, with custom routing strategies (task routing, SMEAR-style routing, … )
3. maintain a collection of experts on HuggingFace, which can later be used to transfer to new tasks, or update existing ones, brining us one step closer to seamless continual development of adapters. 

## Quick Demo

1. Training Experts, and adding them to the library

```python
# Create e.g. LoRA wrapped LLM
expert_trainer = ExpertTrainer(**vars(args))

print(expert_trainer)
"""
You will see something like 
...
(k_proj): LoRA(
   (layer): Linear(in_features=768, out_features=768, bias=False)
)
...
"""

# Train it
pl_trainer = pl.Trainer(....)
ckpt_callback = ... 
pl_trainer.train(expert_trainer, ckpt_callback, ...)
expert_ckpt = ckpt_callback.gest_model_path

# After training, upload it to huggingface
expert_lib = HFExpertLibrary('<your_hf_usename>/<my_lora_library>')
library.add_expert(load_expert(expert_ckpt), expert_name=<task_name>)

```

1. Loading experts from the library, and using them to transfer to new tasks 

```python
# Create a Multi Expert Model (MoE like model which can hold multiple expert instances)
moe = MultiExpertModel(**vars(args))

# Instantiate Our Library 
expert_lib = HFExpertLibrary('<your_hf_usename>/<my_lora_library>')
moe.add_experts_from_library(library)

print(moe)
"""
(k_proj): LoRAExpertContainer(
  (layer): Linear(in_features=768, out_features=768, bias=False)
  (selector): TaskNameSelector()
    (experts): ModuleDict(
        (expert_task_a): LoRA(
            (layer): Linear(in_features=768, out_features=768, bias=False)
         )
         (expert_task_b): LoRA(
            (layer): Linear(in_features=768, out_features=768, bias=False)
         )
      )
   )
"""
# Train it
pl_trainer = pl.Trainer(....)
pl_trainer.train(moe, ckpt_callback, ...)
expert_ckpt = ckpt_callback.gest_model_path

# After training, you can still upload this new task to HF 
# [TODO: how to convert this to an "uploadable" expert ?
expert_lib = HFExpertLibrary('<your_hf_usename>/<my_lora_library>')
library.add_expert(load_expert(expert_ckpt), expert_name=<task_name>)
```

# Important Abstractions

We will describe below some key object classes that will be useful when developing and training experts. 

### Adapter-style Classes

1. **Adapter** : base class for modifiying a base model. Describe key args like `model_modifier` and such … . This includes `LoRA`, `PrefixTuning`, …
2. **Expert :**  this class “wraps” the `Adapter` class, encapsulating both the adapter weights, as well as the required `Config` for said expert and the params used during training
3. **ExpertContainer** : this class contains a collection of `Expert` and a routing mechanism (called `Selector` described below), and takes care of routing the inputs and aggregating the outputs across its collection of experts. 

### Model Wrapper (Lightning) Classes

1. **ExpertTrainer** : is your base pytorch lightning wrapper taking care of (i) backbone model creation, (ii) adapter / expert insertion, and (iii) a potential routing mechanism (from the `Selector` class)
2. **MultiExpertModel** : generalizes `ExpertTrainer` to multiple experts; that is, it contains `ExpertContainers`, and can be used for inference across multiple experts, (and also to train a router ?)
3. **MoeTrainer** : TODO 
4. **RoutedMultiExpertModel** : TODO
5. **MultiExpertModelRanker:** TODO

### Routers

1. `Selector` (in `expert_containers` ) : TODO (please describe what goes in the selectors, and the `Output` classes. What info is needed to create new selectors ?

# TODO:

1. How do update an existing expert ? A code snippet would be very useful.
2. Regarding the code, there seems to be a lot of redundancies and deprecated methods. Let’s try to schedule some cleanup time.
