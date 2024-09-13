

## Generate data with gpt-4o-mini

First, set environment variables OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION. Then if you want to generate summaries:

```python generate_for_wikipedia.py --model gpt-4o-mini --output_path wiki_summ_gpt-4o-mini --use_prompts summary```

Else, if summaries and qa are needed:

```python generate_for_wikipedia.py --model gpt-4o-mini --output_path wiki_summ-qa_gpt-4o-mini --use_prompts summary,qa```

If we want to use a specific model:

```python generate_for_wikipedia.py --model microsoft/Phi-3-medium-4k-instruct --output_path wiki_summ_phi-3_medium --use_prompts summary```

This will create a HF dataset, that can be loaded with `datasets.load_from_disk`. The dataset contains the following columns:

```
['input', 'outputs', 'subject', 'type']
```

`input` is the chunk of text, `outputs` is a list of generations, `subject` is the corresponding `mmlu` subject, `type` is wheter it's `summary` or `qa`.

To read the dataset, we can just use `KMDataloader`.

To train the KMs, just use `python train_km.py -c phi-3.json -k dataset=... -k finetune_task_name=MMLU_SUBJECT expert_name=MMLU_SUBJECT`
