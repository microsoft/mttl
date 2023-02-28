# MTTL

MTTL - Multi-Task Transfer Learning

## Setup

Install Python packages:

`pip install -r requirements.txt`

_The package `promptsource` currently requires Python 3.7. Alternative versions require local installations (see their [documentation](https://github.com/bigscience-workshop/promptsource#setup))._

Download the datasets:

`bash scripts/create_datasets.sh`

## Multi-task Pre-training

The general command:

`python pl_train.py -c $CONFIG_FILES -k $KWARGS`

Multiple `CONFIG_FILES` can be concatenated as `file1+file2`. To modify defaults, `KWARGS` can be expressed as `key=value`.

## Test Fine-Tuning

To perform finetuning for a test task, use the script `pl_finetune.py`

## Hyper-parameter Search for Test Fine-Tuning

To perform an hyperparameter search for a test task, use the script `pl_finetune_tune.py`.
The script will just call the functions in `pl_finetune.py` in a loop. The script itself defines hp ranges for different fine-tuning types.

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
