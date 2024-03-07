# Wiki Expert

The `wiki-experts` library enables users to

1. train different kinds of adapters (LoRA, soft prompts, …) over LLMs
2. build MoE models from these adapters, with custom routing strategies (task routing, SMEAR-style routing, … )
3. maintain a collection of experts on HuggingFace or Azure Blob Storage, which can later be used to transfer
to new tasks, or update existing ones, brining us one step closer to seamless continual development of adapters.

## Quick Demo

1. Training Experts, and adding them to the library

```python
# Create e.g. LoRA wrapped LLM
expert_trainer = ExpertModel(**vars(args))

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

1. **ExpertModel** : is your base pytorch lightning wrapper taking care of (i) backbone model creation, (ii) adapter / expert insertion, and (iii) a potential routing mechanism (from the `Selector` class)
2. **MultiExpertModel** : generalizes `ExpertModel` to multiple experts; that is, it contains `ExpertContainers`, and can be used for inference across multiple experts, (and also to train a router ?)
3. **MoeModel** : TODO


### Routers

1. `Selector` (in `expert_containers` ) : TODO (please describe what goes in the selectors, and the `Output` classes. What info is needed to create new selectors ?


## ExpertLibrary Backend


Wiki-experts supports HuggingFace and Azure Blob Storage for storing and retrieving experts. Use the `HFExpertLibrary` for HuggingFace integration and `BlobExpertLibrary` for Azure Blob Storage. Alternatively, you can use the `ExpertLibrary.get_expert_library` to get the appropriate library instance based on the repository id. In this case, the repository id should follow the format `"hf://<user_id_>/<lib_id>"` for HuggingFace and `"az://<storage_account>/<lib_id>"` for Azure Blob Storage.
You can also use the `destination_id` parameter to create a copy of any kind of the library.

```python
# For HuggingFace
token = "<your_hf_token>"
repo_id = "<your_hf_usename>/<my_lora_library>"
hf_export_lib = HFExpertLibrary(repo_id, token=token)
# or
repo_id = "hf://<your_hf_usename>/<my_lora_library>"
hf_export_lib = ExpertLibrary.get_expert_library(repo_id, token)

# making a copy of the library using the `destination_id` parameter
local_expert_lib = ExpertLibrary.get_expert_library(
   repo_id, token, destination_id="local://my_local_library"
)

# For Azure Blob Storage
token = "<your_sas_token>"
repo_id = "<storage_account>/<my_lora_library>"
blob_export_lib = BlobExpertLibrary(repo_id, token=token)
# or
repo_id = "az://<storage_account>/<my_lora_library>"
blob_export_lib = ExpertLibrary.get_expert_library(repo_id, token)
# or
az_expert_lib = ExpertLibrary.get_expert_library(
   repo_id, token, expert_library_type="az"
) # az, hf, local (filesystem), or virtual (in memory)
```

Note that the `repo_id` for HuggingFace should include your username, while for Azure Blob Storage, it should include the storage account name.

To avoid managing tokens each time you instantiate an `ExpertLibrary`, set the `HF_TOKEN` for HuggingFace or `BLOB_SAS_TOKEN` for Azure Blob Storage as environment variables.


### Getting a HuggingFace API token

1. Navigate to your HuggingFace account settings.
2. Select the "Access Tokens" tab.
3. Click "New token".
4. Copy the generated token.


### Getting an Azure Blob Storage SAS Token

1. Open the Azure Portal.
2. Locate and select your storage account.
3. Click "Shared access signature" from the left-side menu.
4. Ensure 'Blob' is the selected resource type.
5. Check necessary permissions and set an expiration date.
6. Click "Generate SAS".
7. Copy the "SAS token".


### Cache directory

The `HFExpertLibrary` and `BlobExpertLibrary` classes use a cache directory to store the experts.

  - For `HFExpertLibrary`, it defaults to `HF_HUB_CACHE="$HF_HOME/hub"` where `HF_HOME="~/.cache/huggingface"`.

  - For `BlobExpertLibrary`, the default cache directory is set to `BLOB_CACHE_DIR="~/.cache/mttl"`.


You can alter the cache location by setting the corresponding environment variable.


# TODO:

1. How do update an existing expert ? A code snippet would be very useful.
2. Regarding the code, there seems to be a lot of redundancies and deprecated methods. Let’s try to schedule some cleanup time.
