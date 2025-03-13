[![Tests](https://github.com/microsoft/mttl/actions/workflows/tests.yml/badge.svg)](https://github.com/microsoft/mttl/actions/workflows/tests.yml)

# LoRA Soup for GSM-8k 


## train the math skill and code skill. 

For example, we train the code skill based on gptneo model 

```
python train_experts.py -c configs/models/gptneo_125m.json -k dataset=alpaca_code_train_epochs=3 output_dir=debug_alpaca_code
```





## Evaluate the model on GSM

- eval the dense model

1) we generate the python code first:

```
python gsm_evaluator_with_lora_soup.py -k model=EleutherAI/gpt-neo-125m dataset=gsm gsm_template=python max_input_length=2048 max_output_length=128 output_dir=gpt_125m_dense
```
there is a json file in the "gpt_125m_dense" dir. 

2) eval the accuracy of gsm8k

```
python eval_gsm_mttl.py --file=gpt_125m_dense/predict_python_code.jsonl
```

then we got 0.0015


- eval the alpaca_code skill

1) generate the python code

```
python gsm_evaluator_with_lora_soup.py -k model=EleutherAI/gpt-neo-125m dataset=gsm gsm_template=python max_input_length=2048 max_output_length=128 output_dir=gpt_125m_alpaca_code checkpoint=projects/modular_llm
/debug_alpaca_code/best_mode_min_metric_val-loss_value_1.1037_step_1239.ckpt
```

2) eval the gsm8k-hard

we got the same score. It seems the alpaca-code does not help the gpt125m