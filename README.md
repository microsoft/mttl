[![Tests](https://github.com/pclucas14/lucas_mttl/actions/workflows/tests.yml/badge.svg)](https://github.com/pclucas14/lucas_mttl/actions/workflows/tests.yml)


# MTTL

MTTL - Multi-Task Transfer Learning


## Setup

MTTL supports `Python 3.8` and `Python 3.9`. It is recommended to create a virtual environment for MTTL using `virtualenv` or `conda`. For example, with `conda`:

    conda create -n mttl python=3.9
    conda activate mttl

Install the required Python packages:

    pip install -e .



## Multi-Head Adapter Routing

Please ensure that you have navigated to the `projects/mhr` directory before running the Multi-Head Adapter Routing scripts:

    cd projects/mhr


### Data Preparation

Download and prepare the datasets for the experiments using the following script:

    bash datasets/create_datasets.sh


### Environment Variables

Based on your experiments, you may need to export one or more of the following environment variables:

    T0_DATA_DIR:  `data/t0_data/processed` if you ran the `create_datasets.sh`
    NI_DATA_DIR: `data/ni_data/processed` if you ran the `create_datasets.sh`
    XFIT_DATA_DIR: `data/ni_data/processed` if you ran the `create_datasets.sh`
    CHECKPOINT_DIR
    OUTPUT_DIR
    CACHE_DIR
    STORYCLOZE_DIR: path to your downloaded `.csv` files. See [the storycloze official website](https://cs.rochester.edu/nlp/rocstories/)


### Multi-task Pre-training

The general command for pre-training a model is:

    python pl_train.py -c $CONFIG_FILES -k $KWARGS

Multiple `CONFIG_FILES` can be concatenated as `file1+file2`. To modify defaults, `KWARGS` can be expressed as `key=value`.
You can check [scripts/pretrain](scripts/pretrain) for examples.

### Test Fine-Tuning

To perform finetuning for a test task, use the script `pl_finetune.py`

### Hyper-parameter Search for Test Fine-Tuning

To perform an hyperparameter search for a test task, use the script `pl_finetune_tune.py`.
The script will just call the functions in `pl_finetune.py` in a loop. The script itself defines hp ranges for different fine-tuning types.


### Pre-Configured Scripts

Alternatively, you can run the pre-configured scripts from the `scripts` folder. For example:

    bash scripts/mhr_pretrain.sh

### Know Issues
If you run into issues with protoc `TypeError: Descriptors cannot not be created directly.`, you can try to downgrade protobuf to 3.20.*:

    pip install protobuf==3.20.*


## Running Tests

    pip install -e ".[test]"
    pytest -vv tests

## Acknowledgements
We relied on Derek Tam's open-source code for the dataloading and T5 lightning wrapper for our encoder-decoder experiments. You can find the original code [here](https://github.com/r-three/t-few/tree/master).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
