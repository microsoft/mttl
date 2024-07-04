# Towards Modular LLM by Building and Reusing a Library of LoRAs

The code in this folder allows to reproduce the experiments in our paper. Mainly, the code contains scripts that use the MTTL library to:

1. Train different kinds of adapters over LLMs;
2. Build MoE models from these adapters, with custom routing strategies;
3. Maintain a collection of experts, which can later be used to transfer to new tasks or update existing ones.

## Setup

Before starting, make sure to install all the requirements going with MTTL. MTTL supports `Python >=3.8, <3.12`. Create a virtual environment using `virtualenv` or `conda`, then install the required Python packages from the root directory of this repository:

```bash
conda create -n mttl python=3.11
conda activate mttl

pip install -e .
```

Alternatively:

```
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Dataset Preparation

First of all, download and prepare [FLANv2](https://github.com/google-research/FLAN/tree/main/flan/v2) dataset. We limit each task to having 10000 examples for computational reasons. We provide a simple script to do all the preprocessing as below:

```bash
python cli_dataset_create.py flan --dataset_library_id=local://modular_artifacts/flan-flat
```


## Training a Private Library

A *private* library consists of one expert per task in Flan. To train one expert starting from Phi-2, we can use the following command:

```bash
python train_experts_main.py -c configs/models/phi-2_hf.json \
  -k finetune_task_name=ai2_arc_ARC_Easy_1_0_0 \
  output_dir=arc_easy_expert/ \
  dataset=local://modular_artifacts/flan-flat \
  library_id=local://modular_artifacts/library \
  expert_name=arc_easy
```

The expert will be automatically added to an *Expert Library* stored under `modular_artifacts/library`. To know more about the Expert Library concept, continue reading :).

We provide a bash script that loops over all Flan tasks and trains one expert on each:

```bash
export LIBRARY_PATH=local://modular_artifacts/library
export DATASET_PATH=local://modular_artifacts/flan-flat
bash train_private_library.sh
```

To start, you can run `train_private_library_fast.sh` which trains only 2 experts using a small LM (gpt-neo 125M).

After this, to analyze the content of your expert library, you can use the script in `mttl/cli/show_library.py` by providing the path to the library. 

## Training an MBC library

To train an MBC library, we need to cluster a private library. To do so:

```bash
python run_mbc_clustering.py -k \
  library_id=local://modular_artifacts/library \
  num_clusters=10 \
  output_file=modular_artifacts/mbc_10.json
```

The file `mbc_10.json` will contain the task names falling into each cluster. These task names can then be used to train experts by just passing `finetune_task_name=task_name1,task_name2` to the `train_experts_main.py` script.


## Evaluating the Modular LLM

Once we built a library, we can load it into the base model and apply a given merging or routing mechanism, such as Arrow. To evaluate the resulting modular LLM on, for example, arc-easy, you can run:  

```bash
python eval_library.py \
  -k output_dir=an_expert_eval/ \
  library_id=local://modular_artifacts/library \
  pipeline_eval_tasks='arc-easy' \
  merge_or_route='uniform'
```

`merge_or_route='uniform'` means that we will just uniformly average all the experts in the library before performing inference. To run `Arrow`, use `merge_or_route='arrow'` instead:

```bash
python eval_library.py \
  -k output_dir=an_expert_eval/ \
  library_id=local://modular_artifacts/library \
  pipeline_eval_tasks='arc-easy' \
  merge_or_route='arrow' \
  topk=4
```

At first, this will compute Arrow prototypes (thus will be a bit slower) but then the prototypes will be stored inside the library as additional artifacts, therefore subsequent calls will be much faster.

## Additional Documentation around Expert Library

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
