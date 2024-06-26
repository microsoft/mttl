# Expert Library

The MTTL Expert Library enables:

1. Train different kinds of adapters over LLMs;
2. Build MoE models from these adapters, with custom routing strategies;
3. Maintain a collection of experts, which can later be used to transfer to new tasks or update existing ones.


## Setup

MTTL supports `Python >=3.8, <3.12`. Create a virtual environment using `virtualenv` or `conda`, then install the required Python packages:

```bash
conda create -n mttl python=3.11
conda activate mttl

pip install -e .
```


## Expert Library

### Important Abstractions

#### **`Expert` class**

The `Expert` class encapsulates information about the modules, aka experts, represented with LoRA adapters in the paper. Each expert instance contains expert’s configuration (stored in `expert_info` attribute) and the state dictionary of expert’s weights (stored in `_expert_weights` attribute). It provides methods to manage and retrieve this information, ensuring compatibility with different versions of the model's training and configuration data.

#### **`ExpertInfo` class**

The `ExpertInfo` class encapsulates metadata and configuration information for a model's expert. This includes the expert's name, name of the task expert was trained on, configuration details, and the associated base model name. It provides methods to create instances from dictionaries and to convert instances back into dictionaries. Expert instances are stored in **`ExpertLibrary`** as detailed next.

#### **`ExpertLibrary` class**

Central to our research is creating and handling collections of expert models tailored to specific tasks. To aid research and development, we’ve created an `ExpertLibrary` class, which allows you to load trained expert models, and upload new experts in a straightforward way.`ExpertLibrary` provides methods for adding, retrieving, listing, and removing experts within the library. The library can interface with different storage backends such as local filesystem, Azure Blob Storage, and Hugging Face Hub.
the hub.

### ExpertLibrary Backend

Wiki-experts supports Hugging Face, Azure Blob Storage, Local and Virtual backends for storing and retrieving experts. Use  `ExpertLibrary.get_expert_library` to get the appropriate library instance based on a repository id (`repo_id (str)`). The repository id is formed by a prefix that identify the backend type, plus the expert library location. You can also use the `destination_id` parameter to create a copy of any kind of the library.

#### Prefixes

- `hf://` - Hugging Face Hub: Indicates the library is stored in the Hugging Face Hub.
- `az://` - Azure Blob Storage: Indicates the library is stored in Azure Blob Storage.
- `local://` - Local Filesystem: Indicates the library is stored on the local filesystem.
- `virtual://` - Virtual Local Library: Indicates a temporary library stored in memory.

#### Repository id format

When specifying repository IDs, use the following formats:
- For Hugging Face: `"hf://<user_id>/<lib_id>"`
- For Azure Blob Storage: `"az://<storage_account>/<lib_id>"`
- For local and virtual locations: `"[local|virtual]://<path_to_library>"`, where `<path_to_library>` refers to either absolute or relative paths. Note that virtual locations will not be written to the filesystem.

```python
# Loading an expert library using az, hf, local (filesystem), or virtual (in memory)

# HuggingFace
token = "<your_hf_token>"
repo_id = "hf://<your_hf_usename>/<my_lora_library>"
hf_export_lib = ExpertLibrary.get_expert_library(repo_id, token)
# or
repo_id = "<your_hf_usename>/<my_lora_library>"
hf_export_lib = HFExpertLibrary(repo_id, token=token)

# Azure Blob Storage
token = "<your_sas_token>"
repo_id = "az://<storage_account>/<my_lora_library>"
blob_export_lib = ExpertLibrary.get_expert_library(repo_id, token)
# or
repo_id = "<storage_account>/<my_lora_library>"
blob_export_lib = BlobExpertLibrary(repo_id, token=token)

# making a copy of the library using the `destination_id` parameter
local_expert_lib = ExpertLibrary.get_expert_library(
   repo_id, token, destination_id="local://my_local_library"
)
```

### ExpertLibrary Basic Usage:

```python
# Create an ExpertLibrary instance
expert_lib = ExpertLibrary(
    repo_id="my_repo", token="my_token", create=True
)

# Add an expert to the library
expert_info = ExpertInfo(
    expert_name="example_expert", expert_task_name="example_task"
)
expert = Expert(expert_info=expert_info)
expert_lib.add_expert(expert)

# Retrieve an expert from the library
retrieved_expert = expert_lib.get_expert("example_expert")

# Remove an expert from the library
expert_lib.remove_expert("example_expert")
```

### Dataset Preparation

Download and prepare [FLANv2](https://github.com/google-research/FLAN/tree/main/flan/v2) dataset:

```bash
python projects/modular_llm/cli_dataset_create.py flan --dataset_library_id=local://mttldata/flan-flat
```

To cite FLAN. please use:

```
@article{longpre2023flan,
  title={The Flan Collection: Designing Data and Methods for Effective Instruction Tuning},
  author={Longpre, Shayne and Hou, Le and Vu, Tu and Webson, Albert and Chung, Hyung Won and Tay, Yi and Zhou, Denny and Le, Quoc V and Zoph, Barret and Wei, Jason and others},
  journal={arXiv preprint arXiv:2301.13688},
  year={2023}
}
```

### Training Experts

```bash
python train_experts_main.py -c configs/wiki-mmlu/gptneo_125m_flan.json -k finetune_task_name=ai2_arc_ARC_Easy_1_0_0 num_train_epochs=1 output_dir=an_expert/ library_id=local://mttldata/mttladapters-predictor pipeline_eval_tasks='arc-easy' expert_name=predictor
```

### Evaluating Experts

```bash
python eval_library.py -k output_dir=an_expert_eval/ library_id=local://mttldata/mttladapters-predictor pipeline_eval_tasks='arc-easy' merge_or_route='uniform'
```
