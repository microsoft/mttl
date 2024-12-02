# Mixture of Latent experts using Tensor Product

This is the offical code for the paper: Mixture of Latent experts using Tensor Product(TMLR2024 accepted). 

## Setup

Install Python packages:

`pip install -r requirements.txt`

_The package `promptsource` currently requires Python 3.7. Alternative versions require local installations (see their [documentation](https://github.com/bigscience-workshop/promptsource#setup))._

Download the datasets:

`bash scripts/create_datasets.sh`

## Multi-task Pre-training

The general command:
- pretrain the lora:

```
python pl_train.py -c t0/3b.json+t0/poly_lora_sota.json+t0/pretrain.json -k output_dir=pretrain_lora n_skills=1

```

- pretrain the poly:

```
python pl_train.py -c t0/3b.json+t0/poly_lora_sota.json+t0/pretrain.json -k output_dir=pretrain_poly_lora checkpoint=pretrain_poly_lora
```

- pretrain the tensorpoly:

```
python pl_train.py -c t0/3b.json+t0/tensorpoly_lora_sota.json+t0/pretrain.json -k output_dir=pretrain_tensorpoly_lora_order_2 order=2
```

## Fine-Tuning

To perform finetuning for a test task, use the script `pl_finetune.py`

```
for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=pretrain_tensororderpoly_lora_order_2 \
    output_dir=finetune_tensororderpoly_lora_order_2/${dataset} order=4
done
```
## Few-shot Evaluation

```
python get_metrics.py files finetune_tensororderpoly_lora_order_4 --dataset=t0
```
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
