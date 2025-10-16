<div align="center">
    <img src="assets/km.gif" height=200 alt="Knowledge Modules Animation"/>

**Condensing long document knowledge into LoRAs**


[![GitHub](https://img.shields.io/github/license/microsoft/mttl)](https://img.shields.io/github/license/microsoft/mttl)
[![arXiv](https://img.shields.io/badge/arXiv-2503.08727-b31b1b.svg)](https://arxiv.org/abs/2503.08727)

</div>


**What is this?** This repository provides code for training Knowledge Modules (KMs), which are lightweight LoRA adapters that store document level information. They are trained *offline* and can match the performance of Retrieval-Augmented Generation (RAG) approaches. They are trained using a combination of (i) synthetic data generation and (ii) Deep Context Distillation. 


**Table of contents**
- [Setup](#setup)
- [Running Knowledge Modules](#running-knowledge-modules)
  - [Step 1: Synthesize training data](#step-1-synthesize-training-data)
  - [Step 2: Train the Knowledge Modules on the document's synthetized data](#step-2-train-a-knowledge-module-for-each-document-with-deep-context-distillation)
  - [Step 3: Training a Task-Level LoRA / Knowledge Extractor](#step-3-training-a-task-level-lora--knowledge-extractor)
- [Acknowledgments and Citation](#acknowledgments-and-citation)


## Setup

**Step 1:** Clone the repository and install the Python package.

```bash
git clone https://github.com/microsoft/mttl && cd mttl
python -m pip install -e .
python -m pip install vllm
python -m pip install tenacity
python -m pip install langchain_text_splitters
cd projects/kms
```

**Step 2:** (Optional) Set up environment variables. 
You will need to set these only if using API calls to OpenAI for the data generation step

```bash
export OPENAI_API_KEY=openai_api_key
export OPENAI_BASE_URL=openai_base_url
export OPENAI_API_VERSION=openai_api_version
```

If you use HuggingFace, make sure `HF_TOKEN` is set. 

## Running Knowledge Modules 

Learning a Knowledge Module for a given document is a two step process. We first split a document into chunks and generate synthetic data (e.g. summaries, or question-answer pairs) for each document chunk. Then, we train a LoRA adapter using a Deep Context Distillation Objective. 


### Step 1: Synthesize training data

To generate the synthetic data, we use the `generate_for_dataset.py` file. The script genreates a `DatasetAugmenter` class, which takes care of the data generation. It uses a `GenerationTask` (e.g. a `QATask(GenerationTask)`) to define the prompts used for generation. You should be able to extend this should you want additional tasks other than QA or Summary generation. Here are some notable args for `generate_for_dataset.py` : 


```python
    # which model to use (use a HF model identifier, or a path to a local checkpoint)
    model: str = None
    
    # use a HF dataset identifier, or a path to a local file with the `local://` prefix, e.g. `local://my/local/path
    dataset: str = None 
    
    # We support "narrativeqa", "quality", "wiki_top_20". This is used to know how to format / standardize the document. If you have a custom dataset you should add an option here. 
    dataset_type: str = "narrativeqa"

    # Size of a document chunk
    block_size: int = 2048

    # Max Size of the synthesized data (e.g. the summary of the q-a paris)
    max_continuation_length: int = 768

    # How many generations to do per document chunk ? 
    num_generations: int = 16
    
    # What synthetic data to generate (comma separated )
    use_prompts: str = "summary,qa"

    # Where to save the data 
    output_path: str = "/tmp/"

    # Push to HF if provided a `username/dataset_name` string
    push_to_hub: str = None

    # How to do the generations ("vllm" or "oai" supported)
    model_type: str = "vllm"
```

Example command : 
```bash
python generate_for_dataset.py -k model="meta-llama/Llama-3.1-8B-Instruct" dataset=sordonia/quality_sanitized dataset_type=quality  push_to_hub=pclucas14/quality_llama_8B
```

Make sure the [VLLM](https://github.com/vllm-project/vllm) is installed, as we use it to speed-up the generation process!


### Step 2: Train a Knowledge Module for each Document with deep-context-distillation

We use the script `train_km_simple.py` to train the knowledge modules. In the `mttl` repository, we use the following command line setup. You can pass in config files (yaml) that set specific variables to specific values. In this project, you will find all the training configs in `configs/train` (and similarly `configs/eval` for eval). Here's an example on how to reproduce the llama 8B experiment on the `NarrativeQA` dataset, on the document in the dataset with the document id `350c0f8265c2d1183fd9a7e2a92c748998ac8775`

```bash
python train_km_simple.py -c train/llama-nqa-sdcd-fp-fixed-right -k finetune_task_name=350c0f8265c2d1183fd9a7e2a92c748998ac8775
```

**NOTE:** If you don't specify a value for `finetune_task_name`, you will train a KM on all the documents in your dataset.

**NOTE:** When running our experiments, we would queue up 1 of such command, for each of the documents in our dataset. 

Notice here that we use `-c` to pass in a config, and `-k` to overwrite any argument from the command line. You can chain multiple `-k` calls to overwrite multiple arg values. 


### Step 3: Training a Task-Level LoRA / Knowledge Extractor

This is done via the `train_qa.py` script. You can check the script for specific arguments for additional memory efficiency. 

## Additional Questions
This readme is a high-level overview of the repository. Feel free to raise an issue or to send us an email if you have any additional questions.


## Acknowledgments and Citation
Many other teams came out with similar works in this space! You should definitely check their work out. Here are some of them 
1. The Cartridges paper, and their excellent [blog post](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges) and [research code](https://github.com/HazyResearch/cartridges/blob/25ac7a9246fad9306171d1638c2b8e9a6bc0d825/README.md) (from which this README was inspired)
2. The [Self-Distillation paper](https://arxiv.org/abs/2412.14964)

If you used this work, please consider citing the paper!

```bibtex
@inproceedings{
caccia2025training,
title={Training Plug-and-Play Knowledge Modules with Deep Context Distillation},
author={Lucas Caccia and Alan Ansell and Edoardo Ponti and Ivan Vuli{\'c} and Alessandro Sordoni},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=ghyyHZYORi}
}
```
